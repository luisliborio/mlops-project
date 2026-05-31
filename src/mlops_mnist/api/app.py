from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
import mlflow.tensorflow
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Summary, generate_latest

from mlops_mnist.app_reset import _reset_mlflow_runtime_state
from mlops_mnist.utils.runtime import (
    load_runtime_config,
    resolve_runtime_port,
    resolve_tracking_uri,
)
from mlops_mnist.utils.mlflow_compat import (
    prepare_local_tracking_uri,
    resolve_runtime_model_artifact_path,
    resolve_runtime_write_path,
)


MODEL_ALIAS = "production"

HTTP_REQUESTS_TOTAL = Counter(
    "mnist_api_http_requests_total",
    "Total HTTP requests served by the API.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "mnist_api_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
HTTP_ERRORS_TOTAL = Counter(
    "mnist_api_errors_total",
    "Total API requests that ended in an error.",
    ["method", "path"],
)
PREDICTION_CONFIDENCE = Summary(
    "mnist_api_prediction_confidence",
    "Prediction confidences observed by the API.",
)
PREDICTED_CLASS_TOTAL = Counter(
    "mnist_api_predicted_class_total",
    "Predicted digit classes served by the API.",
    ["digit"],
)
LAST_MEAN_CONFIDENCE = Gauge(
    "mnist_api_last_mean_confidence",
    "Mean confidence of the most recent prediction batch.",
)
LAST_BATCH_SIZE = Gauge(
    "mnist_api_last_batch_size",
    "Size of the most recent prediction batch.",
)
MODEL_INFO = Gauge(
    "mnist_api_model_info",
    "Currently loaded production model.",
    ["model_name", "alias", "version", "candidate_name"],
)


# Validate the request body sent to the prediction endpoint.
class PredictRequest(BaseModel):
    instances: list[list[list[float]]] = Field(..., description="Batch of 28x28 grayscale images.")
    source: str | None = Field(default=None, description="Optional source label for request logging.")


# Keep the loaded model and request logging in one place.
class InferenceService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.params = load_project_params(repo_root / "params.yaml")
        self.runtime_config = _load_runtime_config(repo_root)
        self.tracking_uri = _resolve_tracking_uri(
            repo_root,
            self.runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        )
        self.tracking_uri = prepare_local_tracking_uri(repo_root, self.tracking_uri)
        self.logs_root = resolve_runtime_write_path(repo_root, self.runtime_config.get("LOGS_DIR", "logs"))
        self.prediction_log_path = self.logs_root / "predictions" / "prediction_events.jsonl"
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.registered_model_name = self.params["project"]["registered_model_name"]
        self.model_alias = MODEL_ALIAS
        self.model: Any | None = None
        self.model_version: str | None = None
        self.model_loaded_at: str | None = None
        self.source_training_run_id: str | None = None
        self.source_candidate_name: str | None = None
        self._model_info_labels: dict[str, str] | None = None

    def load_model(self) -> None:
        # Load the model version currently assigned to the production alias.
        _reset_mlflow_runtime_state()
        self.tracking_uri = prepare_local_tracking_uri(
            self.repo_root,
            _resolve_tracking_uri(
                self.repo_root,
                self.runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            ),
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        if self._model_info_labels is not None:
            MODEL_INFO.labels(**self._model_info_labels).set(0)
        try:
            version_info = client.get_model_version_by_alias(self.registered_model_name, self.model_alias)
        except MlflowException as exc:
            if _is_expected_missing_model_error(exc):
                self._mark_model_unavailable()
                return
            raise

        self.model_version = str(version_info.version)
        self.source_training_run_id = version_info.tags.get("source_training_run_id")
        self.source_candidate_name = version_info.tags.get("candidate_name")
        self.model = mlflow.tensorflow.load_model(
            f"models:/{self.registered_model_name}/{self.model_version}"
        )
        self.model_loaded_at = _utc_now()
        self._model_info_labels = {
            "model_name": self.registered_model_name,
            "alias": self.model_alias,
            "version": self.model_version or "unknown",
            "candidate_name": self.source_candidate_name or "unknown",
        }
        MODEL_INFO.labels(
            model_name=self.registered_model_name,
            alias=self.model_alias,
            version=self.model_version or "unknown",
            candidate_name=self.source_candidate_name or "unknown",
        ).set(1)

    def _mark_model_unavailable(self) -> None:
        # Clear cached model details when no production model is available.
        self.model = None
        self.model_version = None
        self.model_loaded_at = None
        self.source_training_run_id = None
        self.source_candidate_name = None
        self._model_info_labels = None

    def predict(self, request_payload: PredictRequest, client_host: str | None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Production model is not loaded")

        # Prepare the batch and run inference.
        prepared_images = _prepare_images(request_payload.instances)
        start_time = time.perf_counter()
        probabilities = self.model.predict(prepared_images, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Summarize model outputs for metrics and the API response.
        predicted_labels = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        request_id = str(uuid4())
        prediction_rows = []

        for confidence in confidences:
            PREDICTION_CONFIDENCE.observe(float(confidence))
        for label in predicted_labels:
            PREDICTED_CLASS_TOTAL.labels(digit=str(int(label))).inc()
        LAST_BATCH_SIZE.set(len(predicted_labels))
        LAST_MEAN_CONFIDENCE.set(float(np.mean(confidences)))

        # Build one response row per image.
        for index in range(len(predicted_labels)):
            prediction_rows.append(
                {
                    "index": index,
                    "predicted_label": int(predicted_labels[index]),
                    "confidence": float(confidences[index]),
                    "probabilities": [float(value) for value in probabilities[index].tolist()],
                }
            )

        # Save the request summary for monitoring and later analysis.
        log_row = {
            "timestamp": _utc_now(),
            "request_id": request_id,
            "source": request_payload.source,
            "client_host": client_host,
            "batch_size": len(prediction_rows),
            "latency_ms": latency_ms,
            "model_name": self.registered_model_name,
            "model_alias": self.model_alias,
            "model_version": self.model_version,
            "source_training_run_id": self.source_training_run_id,
            "source_candidate_name": self.source_candidate_name,
            "predicted_labels": [row["predicted_label"] for row in prediction_rows],
            "confidences": [row["confidence"] for row in prediction_rows],
            "mean_confidence": float(np.mean(confidences)),
        }
        _append_jsonl(self.prediction_log_path, log_row)

        return {
            "request_id": request_id,
            "model_name": self.registered_model_name,
            "model_alias": self.model_alias,
            "model_version": self.model_version,
            "batch_size": len(prediction_rows),
            "mean_confidence": float(np.mean(confidences)),
            "predictions": prediction_rows,
        }

    def health(self) -> dict[str, Any]:
        # Report the current serving status and loaded model details.
        return {
            "status": "ok" if self.model is not None else "unavailable",
            "tracking_uri": self.tracking_uri,
            "model_name": self.registered_model_name,
            "model_alias": self.model_alias,
            "model_version": self.model_version,
            "model_loaded_at": self.model_loaded_at,
            "source_training_run_id": self.source_training_run_id,
            "source_candidate_name": self.source_candidate_name,
            "prediction_log_path": str(self.prediction_log_path.relative_to(self.repo_root)),
        }


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the inference service once when the API starts.
    repo_root = Path(__file__).resolve().parents[3]
    service = InferenceService(repo_root)
    service.load_model()
    app.state.service = service
    yield


app = FastAPI(
    title="MNIST Production Inference API",
    version="7.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    # Measure request latency and count errors for all endpoints except /metrics.
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500
    error_recorded = False
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        error_recorded = True
        HTTP_ERRORS_TOTAL.labels(method=request.method, path=path).inc()
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=request.method, path=path).observe(elapsed)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status_code=str(status_code),
        ).inc()
        if status_code >= 400 and not error_recorded:
            HTTP_ERRORS_TOTAL.labels(method=request.method, path=path).inc()


@app.get("/health")
def healthcheck(request: Request) -> dict[str, Any]:
    # Return a quick readiness view for the API.
    service = _get_service(request)
    return service.health()


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    # Return the current production model metadata.
    service = _get_service(request)
    return service.health()


@app.get("/metrics")
def metrics() -> Response:
    # Expose Prometheus metrics in the expected text format.
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    # Run inference and translate service errors into HTTP responses.
    service = _get_service(request)
    try:
        return service.predict(payload, client_host=request.client.host if request.client else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/reload-model")
def reload_model(request: Request) -> dict[str, Any]:
    # Reload the production alias without restarting the API process.
    service = _get_service(request)
    service.load_model()
    return service.health()


def main(argv: list[str] | None = None) -> int:
    # Allow the API to be started directly from the command line.
    parser = argparse.ArgumentParser(description="Run the MNIST FastAPI inference service.")
    parser.add_argument("--host", default=None, help="Optional API host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional API port override.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    runtime_config = _load_runtime_config(repo_root)
    host = args.host or runtime_config.get("API_HOST", "0.0.0.0")
    port = args.port or resolve_runtime_port(runtime_config, "API_PORT", 8000)
    uvicorn.run(app, host=host, port=port)
    return 0


def _get_service(request: Request) -> InferenceService:
    # Read the service stored during FastAPI startup.
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return service


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    return load_runtime_config(repo_root)


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    return resolve_tracking_uri(repo_root, raw_tracking_uri)


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _prepare_images(instances: list[list[list[float]]]) -> np.ndarray:
    # Validate image shape and normalize pixel values for the model.
    images = np.asarray(instances, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("instances must have shape [batch, 28, 28]")
    if images.shape[1:] != (28, 28):
        raise ValueError(f"expected 28x28 images, got {images.shape[1:]}")

    if float(images.max()) > 1.0:
        images = images / 255.0
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ValueError("image values must be in the range [0, 1] or [0, 255]")

    return images[..., np.newaxis]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    # Append one JSON record per line to keep the log stream simple.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _is_expected_missing_model_error(exc: Exception) -> bool:
    # Treat a missing registered model as an unavailable service, not a crash.
    if not isinstance(exc, MlflowException):
        return False

    message = str(exc).lower()
    return "registered model" in message and "not found" in message


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())