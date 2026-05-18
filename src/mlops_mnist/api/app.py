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
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


MODEL_ALIAS = "production"
METRICS_PLACEHOLDER = "# Prometheus metrics will be added in Day 9\n"


class PredictRequest(BaseModel):
    instances: list[list[list[float]]] = Field(..., description="Batch of 28x28 grayscale images.")
    source: str | None = Field(default=None, description="Optional source label for request logging.")


class InferenceService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.params = load_project_params(repo_root / "params.yaml")
        self.runtime_config = _load_runtime_config(repo_root)
        self.tracking_uri = _resolve_tracking_uri(
            repo_root,
            self.runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        )
        self.logs_root = _resolve_path(repo_root, self.runtime_config.get("LOGS_DIR", "logs"))
        self.prediction_log_path = self.logs_root / "predictions" / "prediction_events.jsonl"
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.registered_model_name = self.params["project"]["registered_model_name"]
        self.model_alias = MODEL_ALIAS
        self.model: Any | None = None
        self.model_version: str | None = None
        self.model_loaded_at: str | None = None
        self.source_training_run_id: str | None = None
        self.source_candidate_name: str | None = None

    def load_model(self) -> None:
        # Load the current production alias from the local MLflow registry.
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        version_info = client.get_model_version_by_alias(self.registered_model_name, self.model_alias)
        self.model_version = str(version_info.version)
        self.source_training_run_id = version_info.tags.get("source_training_run_id")
        self.source_candidate_name = version_info.tags.get("candidate_name")
        self.model = mlflow.tensorflow.load_model(
            f"models:/{self.registered_model_name}@{self.model_alias}"
        )
        self.model_loaded_at = _utc_now()

    def predict(self, request_payload: PredictRequest, client_host: str | None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Production model is not loaded")

        prepared_images = _prepare_images(request_payload.instances)
        start_time = time.perf_counter()
        probabilities = self.model.predict(prepared_images, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        predicted_labels = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        request_id = str(uuid4())
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
    repo_root = Path(__file__).resolve().parents[3]
    service = InferenceService(repo_root)
    service.load_model()
    app.state.service = service
    yield


app = FastAPI(
    title="MNIST Production Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck(request: Request) -> dict[str, Any]:
    service = _get_service(request)
    return service.health()


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    service = _get_service(request)
    return service.health()


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=METRICS_PLACEHOLDER, media_type="text/plain; version=0.0.4")


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    service = _get_service(request)
    try:
        return service.predict(payload, client_host=request.client.host if request.client else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MNIST FastAPI inference service.")
    parser.add_argument("--host", default=None, help="Optional API host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional API port override.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    runtime_config = _load_runtime_config(repo_root)
    host = args.host or runtime_config.get("API_HOST", "0.0.0.0")
    port = args.port or int(runtime_config.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
    return 0


def _get_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return service


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    config: dict[str, str] = {}

    example_path = repo_root / ".env.example"
    env_path = repo_root / ".env"

    if example_path.exists():
        config.update({key: value for key, value in dotenv_values(example_path).items() if value})
    if env_path.exists():
        config.update({key: value for key, value in dotenv_values(env_path).items() if value})

    return config


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    if raw_tracking_uri.startswith("sqlite:////"):
        return raw_tracking_uri

    if raw_tracking_uri.startswith("sqlite:///"):
        database_path = (repo_root / raw_tracking_uri.removeprefix("sqlite:///" )).resolve()
        return f"sqlite:///{database_path.as_posix()}"

    return raw_tracking_uri


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _prepare_images(instances: list[list[list[float]]]) -> np.ndarray:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())