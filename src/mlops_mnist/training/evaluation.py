from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from mlops_mnist.training.reference_data import resolve_training_reference_metadata
from mlops_mnist.utils.mlflow_compat import (
    prepare_local_tracking_uri,
    resolve_runtime_model_artifact_path,
    resolve_repo_bound_path,
    resolve_runtime_write_path,
    serialize_repo_bound_path,
)
from mlops_mnist.utils.runtime import load_runtime_config, resolve_tracking_uri


MODEL_ALIAS = "production"
BASELINE_FREEZE_BATCHES = ["baseline_test", "clean_batch_a"]


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_registered_model(
    params_path: str | Path = "params.yaml",
    batch_name: str | None = None,
    batch_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/evaluation",
    registered_model_name: str | None = None,
    alias: str = MODEL_ALIAS,
    max_samples: int | None = None,
    use_clean_labels: bool = False,
    freeze_baseline: bool = False,
) -> dict[str, Any]:
    # Evaluate the current production alias on one labeled batch or freeze a baseline snapshot.
    params_path = Path(params_path).resolve()
    repo_root = params_path.parent
    params = load_project_params(params_path)
    runtime_config = _load_runtime_config(repo_root)

    # Resolve the registry, data, and report locations.
    tracking_uri = _resolve_tracking_uri(
        repo_root,
        runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    resolved_data_root = _resolve_path(
        repo_root,
        runtime_config.get("DATA_ROOT", "data"),
    )
    resolved_output_dir = resolve_runtime_write_path(repo_root, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_name = registered_model_name or params["project"]["registered_model_name"]

    tracking_uri = prepare_local_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(resolved_model_name, alias)
    loaded_model = mlflow.tensorflow.load_model(
        f"models:/{resolved_model_name}/{version_info.version}"
    )
    production_reference_data = resolve_training_reference_metadata(repo_root, version_info.tags)

    session_dir = resolved_output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate one batch or a baseline pair of batches.
    batch_specs = _resolve_batch_specs(
        data_root=resolved_data_root,
        batch_name=batch_name,
        batch_path=batch_path,
        freeze_baseline=freeze_baseline,
    )
    batch_reports = []

    for current_batch_name, current_batch_path in batch_specs:
        # Score the model and attach registry metadata to the report.
        batch_report = _evaluate_batch(
            model=loaded_model,
            batch_path=current_batch_path,
            batch_name=current_batch_name,
            max_samples=max_samples,
            use_clean_labels=use_clean_labels,
        )
        batch_report.update(
            {
                "registered_model_name": resolved_model_name,
                "alias": alias,
                "model_version": str(version_info.version),
                "source_training_run_id": version_info.tags.get("source_training_run_id"),
                "source_candidate_name": version_info.tags.get("candidate_name"),
                "tracking_uri": tracking_uri,
            }
        )
        batch_report["batch_path"] = serialize_repo_bound_path(repo_root, batch_report["batch_path"])

        report_path = session_dir / f"{current_batch_name}_evaluation.json"
        _write_json(report_path, batch_report)
        batch_report["report_path"] = str(report_path.relative_to(repo_root))
        batch_reports.append(batch_report)

    # Save one summary that later monitoring steps can read back.
    summary_payload = {
        "registered_model_name": resolved_model_name,
        "alias": alias,
        "model_version": str(version_info.version),
        "source_training_run_id": version_info.tags.get("source_training_run_id"),
        "source_candidate_name": version_info.tags.get("candidate_name"),
        "tracking_uri": tracking_uri,
        "freeze_baseline": freeze_baseline,
        "production_reference_data": production_reference_data,
        "reports": batch_reports,
    }

    summary_path = session_dir / (
        "baseline_production_snapshot.json" if freeze_baseline else "evaluation_summary.json"
    )
    _write_json(summary_path, summary_payload)
    summary_payload["summary_path"] = str(summary_path.relative_to(repo_root))
    return summary_payload


def run_training_registry_smoke(
    params_path: str | Path = "params.yaml",
    summary_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/smoke",
    max_samples: int = 32,
) -> dict[str, Any]:
    # Minimal smoke check: latest training summary exists and the production alias can score a small batch.
    params_path = Path(params_path).resolve()
    repo_root = params_path.parent
    selected_summary_path = _resolve_lifecycle_summary(repo_root, summary_path)
    lifecycle_summary = json.loads(selected_summary_path.read_text(encoding="utf-8"))
    evaluation_summary = evaluate_registered_model(
        params_path=params_path,
        batch_name="baseline_test",
        output_dir=output_dir,
        max_samples=max_samples,
    )

    production_run_id = evaluation_summary["source_training_run_id"]
    summary_run_ids = _extract_summary_run_ids(lifecycle_summary)
    smoke_payload = {
        "summary_path": str(selected_summary_path.relative_to(repo_root)),
        "summary_run_ids_found": len(summary_run_ids),
        "production_run_found_in_summary": production_run_id in summary_run_ids,
        "production_run_id": production_run_id,
        "evaluation_summary": evaluation_summary,
    }

    smoke_dir = _resolve_path(repo_root, output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_report_path = smoke_dir / "training_registry_smoke.json"
    _write_json(smoke_report_path, smoke_payload)
    smoke_payload["report_path"] = str(smoke_report_path.relative_to(repo_root))
    return smoke_payload


def main(argv: list[str] | None = None) -> int:
    # Expose model evaluation through a small command-line interface.
    parser = argparse.ArgumentParser(
        description="Evaluate the current production model on labeled MNIST batches.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--batch-name", default=None, help="Named batch under data/processed, such as clean_batch_a.")
    parser.add_argument("--batch-path", default=None, help="Optional explicit path to an .npz batch file.")
    parser.add_argument("--output-dir", default="artifacts/evaluation", help="Folder for evaluation reports.")
    parser.add_argument(
        "--registered-model-name",
        default=None,
        help="Optional MLflow registered model name override.",
    )
    parser.add_argument("--alias", default=MODEL_ALIAS, help="Model alias to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for scored samples.")
    parser.add_argument(
        "--use-clean-labels",
        action="store_true",
        help="Use clean_labels when the batch file contains them.",
    )
    parser.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="Evaluate baseline_test and clean_batch_a together and write a baseline snapshot.",
    )
    args = parser.parse_args(argv)

    summary = evaluate_registered_model(
        params_path=args.params,
        batch_name=args.batch_name,
        batch_path=args.batch_path,
        output_dir=args.output_dir,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        max_samples=args.max_samples,
        use_clean_labels=args.use_clean_labels,
        freeze_baseline=args.freeze_baseline,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    return load_runtime_config(repo_root)


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    return resolve_repo_bound_path(repo_root, raw_path)


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    return resolve_tracking_uri(repo_root, raw_tracking_uri)


def _resolve_batch_specs(
    data_root: Path,
    batch_name: str | None,
    batch_path: str | Path | None,
    freeze_baseline: bool,
) -> list[tuple[str, Path]]:
    # Decide which batch files should be scored in this run.
    if freeze_baseline:
        return [
            (current_batch_name, data_root / "processed" / f"{current_batch_name}.npz")
            for current_batch_name in BASELINE_FREEZE_BATCHES
        ]

    if batch_path is not None:
        resolved_path = resolve_repo_bound_path(data_root.parent, batch_path)
        return [(resolved_path.stem, resolved_path)]

    if batch_name is None:
        raise ValueError("Provide --batch-name, --batch-path, or --freeze-baseline")

    return [(batch_name, data_root / "processed" / f"{batch_name}.npz")]


def _evaluate_batch(
    model: Any,
    batch_path: Path,
    batch_name: str,
    max_samples: int | None,
    use_clean_labels: bool,
) -> dict[str, Any]:
    # Score one labeled batch and summarize the prediction distribution.
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    with np.load(batch_path) as batch:
        images = batch["images"].astype(np.float32)
        labels_key = "clean_labels" if use_clean_labels and "clean_labels" in batch.files else "labels"
        labels = batch[labels_key].astype(np.int64)

    if max_samples is not None:
        images = images[:max_samples]
        labels = labels[:max_samples]

    probabilities = model.predict(images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    return {
        "batch_name": batch_name,
        "batch_path": str(batch_path),
        "labels_key": labels_key,
        "num_examples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "macro_f1": float(f1_score(labels, predicted_labels, average="macro")),
        "mean_confidence": float(np.mean(confidences)),
        "label_distribution": np.bincount(labels, minlength=10).tolist(),
        "predicted_distribution": np.bincount(predicted_labels, minlength=10).tolist(),
    }


def _resolve_lifecycle_summary(repo_root: Path, summary_path: str | Path | None) -> Path:
    # Use the provided summary path or fall back to the latest lifecycle artifact.
    if summary_path is not None:
        resolved = _resolve_path(repo_root, summary_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Lifecycle summary file not found: {resolved}")
        return resolved

    training_root = repo_root / "artifacts" / "training"
    retraining_root = repo_root / "artifacts" / "retraining"
    candidates = sorted(training_root.glob("*/summary.json")) + sorted(
        retraining_root.glob("*/retraining_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "No summary.json files found under artifacts/training or artifacts/retraining."
        )
    return candidates[-1]


def _extract_summary_run_ids(summary_payload: dict[str, Any]) -> set[str]:
    # Collect every run id referenced by a lifecycle summary.
    run_ids = {str(result["run_id"]) for result in summary_payload.get("results", [])}

    promotion = summary_payload.get("promotion")
    if isinstance(promotion, dict) and promotion.get("run_id"):
        run_ids.add(str(promotion["run_id"]))

    production_model = summary_payload.get("production_model")
    if isinstance(production_model, dict) and production_model.get("source_training_run_id"):
        run_ids.add(str(production_model["source_training_run_id"]))

    return run_ids


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # Persist JSON artifacts with stable formatting.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())