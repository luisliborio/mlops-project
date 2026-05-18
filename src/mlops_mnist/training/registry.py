from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from dotenv import dotenv_values
from mlflow.tracking import MlflowClient
from tensorflow import keras


MODEL_ALIAS = "production"
MODEL_READY_POLL_SECONDS = 1.0
MODEL_READY_MAX_POLLS = 30


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def promote_best_model(
    params_path: str | Path = "params.yaml",
    summary_path: str | Path | None = None,
    candidate_name: str | None = None,
    registered_model_name: str | None = None,
    alias: str = MODEL_ALIAS,
    data_root: str | Path | None = None,
    verify_samples: int = 5,
) -> dict[str, Any]:
    # Promote the best saved Keras model into the MLflow registry with an alias.
    params_path = Path(params_path).resolve()
    repo_root = params_path.parent
    params = load_project_params(params_path)
    runtime_config = _load_runtime_config(repo_root)

    tracking_uri = _resolve_tracking_uri(
        repo_root,
        runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    artifact_root = _resolve_artifact_root(
        repo_root,
        runtime_config.get("MLFLOW_ARTIFACT_ROOT", "./artifacts"),
    )
    experiment_name = runtime_config.get(
        "MLFLOW_EXPERIMENT_NAME",
        params["project"].get("experiment_name", params["project"]["name"]),
    )
    resolved_data_root = _resolve_path(
        repo_root,
        data_root or runtime_config.get("DATA_ROOT", "data"),
    )

    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    selected_summary_path = _resolve_summary_path(repo_root, summary_path)
    summary_payload = json.loads(selected_summary_path.read_text(encoding="utf-8"))
    selected_result = _select_result(summary_payload, candidate_name)

    model_path = _resolve_path(repo_root, selected_result["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model file not found: {model_path}")

    resolved_model_name = registered_model_name or params["project"]["registered_model_name"]
    report_dir = repo_root / "artifacts" / "registry" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)
    run_name = f"promote-{selected_result['candidate_name']}-{report_dir.name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "selection_rule": "best_validation_accuracy,test_accuracy,validation_macro_f1",
                "selected_candidate": selected_result["candidate_name"],
                "source_training_run_id": selected_result["run_id"],
                "source_summary": str(selected_summary_path.relative_to(repo_root)),
                "source_model_path": str(model_path.relative_to(repo_root)),
                "registered_model_name": resolved_model_name,
                "registered_alias": alias,
                "verify_samples": verify_samples,
            }
        )
        mlflow.log_metrics(
            {
                "selected_best_validation_accuracy": float(selected_result["best_validation_accuracy"]),
                "selected_test_accuracy": float(selected_result["test_accuracy"]),
                "selected_validation_macro_f1": float(selected_result["validation_macro_f1"]),
                "selected_test_macro_f1": float(selected_result["test_macro_f1"]),
            }
        )

        mlflow.tensorflow.log_model(model, artifact_path="model")
        promotion_report_path = report_dir / "promotion_summary.json"
        verification_path = report_dir / "verification_predictions.json"

        mlflow.log_artifact(str(selected_summary_path), artifact_path="selection")

        registered_version = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=resolved_model_name,
        )
        _wait_for_model_version(client, resolved_model_name, registered_version.version)

        client.set_registered_model_alias(
            name=resolved_model_name,
            alias=alias,
            version=registered_version.version,
        )
        client.set_model_version_tag(
            name=resolved_model_name,
            version=registered_version.version,
            key="source_training_run_id",
            value=str(selected_result["run_id"]),
        )
        client.set_model_version_tag(
            name=resolved_model_name,
            version=registered_version.version,
            key="candidate_name",
            value=str(selected_result["candidate_name"]),
        )

        verification_payload = _verify_registered_model(
            registered_model_name=resolved_model_name,
            alias=alias,
            data_root=resolved_data_root,
            verify_samples=verify_samples,
        )

        promotion_payload = {
            "promotion_run_id": run.info.run_id,
            "registered_model_name": resolved_model_name,
            "registered_version": registered_version.version,
            "alias": alias,
            "tracking_uri": tracking_uri,
            "source_summary": str(selected_summary_path.relative_to(repo_root)),
            "selected_result": selected_result,
            "verification": verification_payload,
        }

        _write_json(promotion_report_path, promotion_payload)
        _write_json(verification_path, verification_payload)
        mlflow.log_artifact(str(promotion_report_path), artifact_path="promotion")
        mlflow.log_artifact(str(verification_path), artifact_path="promotion")

    promotion_payload["report_path"] = str(promotion_report_path.relative_to(repo_root))
    promotion_payload["verification_path"] = str(verification_path.relative_to(repo_root))
    return promotion_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the best trained MNIST model in MLflow and assign a production alias.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path to a training summary.json file. Defaults to the latest one.",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Optional candidate name override. If omitted, the best candidate is selected.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=None,
        help="Optional MLflow registered model name override.",
    )
    parser.add_argument("--alias", default=MODEL_ALIAS, help="Alias to assign in the MLflow registry.")
    parser.add_argument("--data-root", default=None, help="Optional path to the local data directory.")
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="How many baseline test samples to score after promotion.",
    )
    args = parser.parse_args(argv)

    summary = promote_best_model(
        params_path=args.params,
        summary_path=args.summary_path,
        candidate_name=args.candidate,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        data_root=args.data_root,
        verify_samples=args.verify_samples,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    config: dict[str, str] = {}

    example_path = repo_root / ".env.example"
    env_path = repo_root / ".env"

    if example_path.exists():
        config.update({key: value for key, value in dotenv_values(example_path).items() if value})
    if env_path.exists():
        config.update({key: value for key, value in dotenv_values(env_path).items() if value})

    return config


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    if raw_tracking_uri.startswith("sqlite:////"):
        return raw_tracking_uri

    if raw_tracking_uri.startswith("sqlite:///"):
        database_path = (repo_root / raw_tracking_uri.removeprefix("sqlite:///" )).resolve()
        return f"sqlite:///{database_path.as_posix()}"

    return raw_tracking_uri


def _resolve_artifact_root(repo_root: Path, raw_artifact_root: str) -> Path:
    artifact_root = _resolve_path(repo_root, raw_artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def _ensure_experiment(experiment_name: str, artifact_root: Path) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())


def _resolve_summary_path(repo_root: Path, summary_path: str | Path | None) -> Path:
    if summary_path is not None:
        resolved = _resolve_path(repo_root, summary_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Training summary file not found: {resolved}")
        return resolved

    training_root = repo_root / "artifacts" / "training"
    if not training_root.exists():
        raise FileNotFoundError("No training artifacts found. Run the training script first.")

    candidates = sorted(training_root.glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError("No summary.json files found under artifacts/training.")
    return candidates[-1]


def _select_result(summary_payload: dict[str, Any], candidate_name: str | None) -> dict[str, Any]:
    results = list(summary_payload.get("results", []))
    if not results:
        raise ValueError("summary.json does not contain any training results")

    if candidate_name is not None:
        for result in results:
            if result["candidate_name"] == candidate_name:
                return result
        available = ", ".join(result["candidate_name"] for result in results)
        raise ValueError(f"Candidate '{candidate_name}' not found. Available: {available}")

    return max(
        results,
        key=lambda item: (
            float(item["best_validation_accuracy"]),
            float(item["test_accuracy"]),
            float(item["validation_macro_f1"]),
        ),
    )


def _wait_for_model_version(client: MlflowClient, model_name: str, version: str | int) -> None:
    for _ in range(MODEL_READY_MAX_POLLS):
        model_version = client.get_model_version(model_name, str(version))
        status = model_version.status.upper()
        if status == "READY":
            return
        if status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Model version {model_name} v{version} failed to register")
        time.sleep(MODEL_READY_POLL_SECONDS)

    raise TimeoutError(f"Timed out waiting for model version {model_name} v{version} to become READY")


def _verify_registered_model(
    registered_model_name: str,
    alias: str,
    data_root: Path,
    verify_samples: int,
) -> dict[str, Any]:
    baseline_test_path = data_root / "processed" / "baseline_test.npz"
    if not baseline_test_path.exists():
        raise FileNotFoundError(f"Missing baseline test split for verification: {baseline_test_path}")

    with np.load(baseline_test_path) as batch:
        images = batch["images"].astype(np.float32)
        labels = batch["labels"].astype(np.int64)

    sample_images = images[:verify_samples]
    sample_labels = labels[:verify_samples]
    loaded_model = mlflow.tensorflow.load_model(f"models:/{registered_model_name}@{alias}")
    probabilities = loaded_model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    samples = []
    for index in range(len(sample_images)):
        samples.append(
            {
                "index": index,
                "true_label": int(sample_labels[index]),
                "predicted_label": int(predicted_labels[index]),
                "confidence": float(confidences[index]),
            }
        )

    accuracy = float(np.mean(predicted_labels == sample_labels)) if len(sample_labels) else 0.0
    return {
        "registered_model_uri": f"models:/{registered_model_name}@{alias}",
        "checked_samples": len(samples),
        "sample_accuracy": accuracy,
        "samples": samples,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())