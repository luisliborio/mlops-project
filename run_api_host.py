from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Summary, generate_latest

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_mnist.app_reset import _reset_mlflow_runtime_state


MODEL_ALIAS = "production"
RUNTIME_ROOT = Path(".runtime")

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


class PredictRequest(BaseModel):
    instances: list[list[list[float]]] = Field(..., description="Batch of 28x28 grayscale images.")
    source: str | None = Field(default=None, description="Optional source label for request logging.")


class InferenceService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.params = load_project_params(self.repo_root / "params.yaml")
        self.runtime_config = load_runtime_config(self.repo_root)
        self.tracking_uri = prepare_runtime_tracking_uri(self.repo_root, self.runtime_config)
        self.logs_root = ensure_runtime_path(self.repo_root / "logs")
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
        _reset_mlflow_runtime_state()
        self.tracking_uri = prepare_runtime_tracking_uri(self.repo_root, self.runtime_config)
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        if self._model_info_labels is not None:
            MODEL_INFO.labels(**self._model_info_labels).set(0)

        try:
            version_info = client.get_model_version_by_alias(self.registered_model_name, self.model_alias)
        except MlflowException as exc:
            if is_expected_missing_model_error(exc):
                self.mark_model_unavailable()
                return
            raise

        model_artifact_path = prepare_runtime_model_artifact(self.repo_root, version_info.source)

        self.model_version = str(version_info.version)
        self.source_training_run_id = version_info.tags.get("source_training_run_id")
        self.source_candidate_name = version_info.tags.get("candidate_name")
        self.model = mlflow.tensorflow.load_model(
            model_artifact_path.as_posix() if model_artifact_path is not None else version_info.source
        )
        self.model_loaded_at = utc_now()
        self._model_info_labels = {
            "model_name": self.registered_model_name,
            "alias": self.model_alias,
            "version": self.model_version or "unknown",
            "candidate_name": self.source_candidate_name or "unknown",
        }
        MODEL_INFO.labels(**self._model_info_labels).set(1)

    def mark_model_unavailable(self) -> None:
        self.model = None
        self.model_version = None
        self.model_loaded_at = None
        self.source_training_run_id = None
        self.source_candidate_name = None
        self._model_info_labels = None

    def predict(self, request_payload: PredictRequest, client_host: str | None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Production model is not loaded")

        prepared_images = prepare_images(request_payload.instances)
        start_time = time.perf_counter()
        probabilities = self.model.predict(prepared_images, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        predicted_labels = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        request_id = str(uuid4())

        for confidence in confidences:
            PREDICTION_CONFIDENCE.observe(float(confidence))
        for label in predicted_labels:
            PREDICTED_CLASS_TOTAL.labels(digit=str(int(label))).inc()
        LAST_BATCH_SIZE.set(len(predicted_labels))
        LAST_MEAN_CONFIDENCE.set(float(np.mean(confidences)))

        prediction_rows = []
        for index in range(len(predicted_labels)):
            prediction_rows.append(
                {
                    "index": index,
                    "predicted_label": int(predicted_labels[index]),
                    "confidence": float(confidences[index]),
                    "probabilities": [float(value) for value in probabilities[index].tolist()],
                }
            )

        append_jsonl(
            self.prediction_log_path,
            {
                "timestamp": utc_now(),
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
            },
        )

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


def load_runtime_config(repo_root: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    example_path = repo_root / ".env.example"
    env_path = repo_root / ".env"

    if example_path.exists():
        config.update({key: value for key, value in dotenv_values(example_path).items() if value})
    if env_path.exists():
        config.update({key: value for key, value in dotenv_values(env_path).items() if value})

    return config


def resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    if raw_tracking_uri.startswith("sqlite:////"):
        return raw_tracking_uri
    if raw_tracking_uri.startswith("sqlite:///"):
        database_path = (repo_root / raw_tracking_uri.removeprefix("sqlite:///")).resolve()
        return f"sqlite:///{database_path.as_posix()}"
    return raw_tracking_uri


def prepare_runtime_tracking_uri(repo_root: Path, runtime_config: dict[str, str]) -> str:
    tracking_uri = resolve_tracking_uri(
        repo_root,
        runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    database_path = resolve_sqlite_database_path(tracking_uri)
    if database_path is None:
        return tracking_uri

    runtime_database_path = ensure_runtime_path(repo_root / "mlflow.db")
    runtime_database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists() and database_path.resolve() != runtime_database_path.resolve():
        shutil.copy2(database_path, runtime_database_path)
    return f"sqlite:///{runtime_database_path.as_posix()}"


def resolve_sqlite_database_path(tracking_uri: str) -> Path | None:
    if tracking_uri.startswith("sqlite:////"):
        return Path(tracking_uri.removeprefix("sqlite:///"))
    if tracking_uri.startswith("sqlite:///"):
        return Path(tracking_uri.removeprefix("sqlite:///"))
    return None


def ensure_runtime_path(original_path: Path) -> Path:
    try:
        relative_path = original_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative_path = original_path.name if original_path.is_absolute() else original_path
    return (Path.cwd() / RUNTIME_ROOT / relative_path).resolve()


def prepare_runtime_model_artifact(repo_root: Path, model_source: str) -> Path | None:
    if not model_source.startswith("models:/m-"):
        return None

    model_id = model_source.removeprefix("models:/")
    source_path = repo_root / "artifacts" / "models" / model_id / "artifacts"
    if not source_path.exists():
        return None

    runtime_path = ensure_runtime_path(source_path)
    if not runtime_path.exists():
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, runtime_path, dirs_exist_ok=True)

    mlmodel_path = runtime_path / "MLmodel"
    if mlmodel_path.exists():
        original_text = mlmodel_path.read_text(encoding="utf-8")
        lines = original_text.splitlines()
        if lines and lines[0].startswith("artifact_path: "):
            lines[0] = f"artifact_path: {runtime_path.as_uri()}"
            rewritten_text = "\n".join(lines)
            if original_text.endswith("\n"):
                rewritten_text += "\n"
            mlmodel_path.write_text(rewritten_text, encoding="utf-8")

    return runtime_path


def prepare_images(instances: list[list[list[float]]]) -> np.ndarray:
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo_root = Path(__file__).resolve().parent
    service = InferenceService(repo_root)
    service.load_model()
    app.state.service = service
    yield


app = FastAPI(
    title="MNIST Production Inference API (Host Runner)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
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
        HTTP_REQUESTS_TOTAL.labels(method=request.method, path=path, status_code=str(status_code)).inc()
        if status_code >= 400 and not error_recorded:
            HTTP_ERRORS_TOTAL.labels(method=request.method, path=path).inc()


def get_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return service


@app.get("/health")
def healthcheck(request: Request) -> dict[str, Any]:
    return get_service(request).health()


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    return get_service(request).health()


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    service = get_service(request)
    try:
        return service.predict(payload, client_host=request.client.host if request.client else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def is_expected_missing_model_error(exc: Exception) -> bool:
    if not isinstance(exc, MlflowException):
        return False

    message = str(exc).lower()
    return "registered model" in message and "not found" in message


@app.post("/reload-model")
def reload_model(request: Request) -> dict[str, Any]:
    service = get_service(request)
    service.load_model()
    return service.health()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the host-compatible MNIST FastAPI inference service.")
    parser.add_argument("--host", default=None, help="Optional API host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional API port override.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    runtime_config = load_runtime_config(repo_root)
    host = args.host or runtime_config.get("API_HOST", "0.0.0.0")
    port = args.port or int(runtime_config.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())