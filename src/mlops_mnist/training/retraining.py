from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from mlops_mnist.monitoring import analyze_batch_monitoring
from mlops_mnist.training.reference_data import (
    TRAINING_REFERENCE_NAME_TAG,
    TRAINING_REFERENCE_PATH_TAG,
    TRAINING_REFERENCE_SIZE_TAG,
    load_training_reference_data,
    materialize_training_reference_data,
)
from mlops_mnist.training.registry import MODEL_ALIAS
from mlops_mnist.training.trainer import EARLY_STOPPING_PATIENCE
from mlops_mnist.utils.mlflow_compat import (
    prepare_local_tracking_uri,
    resolve_repo_bound_path,
    resolve_runtime_model_artifact_path,
    resolve_runtime_write_path,
)
from mlops_mnist.utils.runtime import load_runtime_config, resolve_artifact_root, resolve_tracking_uri


MODEL_READY_POLL_SECONDS = 1.0
MODEL_READY_MAX_POLLS = 30
BENCHMARK_BATCH_NAME = "baseline_test"
TRAINING_REFERENCE_OUTPUT_DIR = Path("data/processed/training_references")


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_retraining_workflow(
    params_path: str | Path = "params.yaml",
    monitoring_summary_path: str | Path | None = None,
    monitoring_batch_name: str | None = None,
    retraining_batch_name: str | None = None,
    retraining_batch_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/retraining",
    models_dir: str | Path = "models/retraining",
    registered_model_name: str | None = None,
    alias: str = MODEL_ALIAS,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    validation_fraction: float | int | None = None,
    adaptation_test_size: float | int | None = None,
    holdout_fraction: float | None = None,
    use_clean_labels_for_evaluation: bool = True,
    force_retrain: bool = False,
    promote_on_accept: bool = True,
) -> dict[str, Any]:
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
    resolved_model_name = registered_model_name or params["project"]["registered_model_name"]
    base_seed = int(params["project"]["random_seed"])
    resolved_validation_size = (
        validation_fraction
        if validation_fraction is not None
        else params["data"]["baseline"]["validation_size"]
    )
    resolved_adaptation_test_size = (
        adaptation_test_size
        if adaptation_test_size is not None
        else holdout_fraction
        if holdout_fraction is not None
        else params["data"]["baseline"]["adaptation_test_size"]
    )
    acceptance_policy = _resolve_acceptance_policy(params)

    tracking_uri = prepare_local_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    production_version = client.get_model_version_by_alias(resolved_model_name, alias)
    production_uri = f"models:/{resolved_model_name}/{production_version.version}"
    current_training_reference = load_training_reference_data(repo_root, production_version.tags)

    monitoring_summary = _resolve_monitoring_summary(
        repo_root=repo_root,
        params_path=params_path,
        monitoring_summary_path=monitoring_summary_path,
        monitoring_batch_name=monitoring_batch_name,
        retraining_batch_name=retraining_batch_name,
        retraining_batch_path=retraining_batch_path,
    )
    resolved_batch_name, resolved_batch_path = _resolve_retraining_source(
        repo_root=repo_root,
        monitoring_summary=monitoring_summary,
        retraining_batch_name=retraining_batch_name,
        retraining_batch_path=retraining_batch_path,
    )

    session_dir = resolve_runtime_write_path(repo_root, output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    models_root = resolve_runtime_write_path(repo_root, models_dir) / session_dir.name
    models_root.mkdir(parents=True, exist_ok=True)
    summary_path = session_dir / "retraining_summary.json"

    if not force_retrain and not monitoring_summary["should_retrain"]:
        summary = {
            "decision": {
                "accepted": False,
                "action": "keep_production",
                "reason": "monitoring_thresholds_not_met",
                "adaptation_accuracy_gain": 0.0,
                "required_adaptation_accuracy_gain": acceptance_policy["min_adaptation_accuracy_gain"],
                "benchmark_accuracy_change": 0.0,
                "benchmark_accuracy_regression": 0.0,
                "max_allowed_benchmark_accuracy_regression": acceptance_policy["max_benchmark_accuracy_regression"],
            },
            "monitoring_summary": monitoring_summary["summary_path"],
            "retraining_source": {
                "batch_name": resolved_batch_name,
                "batch_path": str(resolved_batch_path.relative_to(repo_root)),
            },
            "production_model": {
                "registered_model_name": resolved_model_name,
                "alias": alias,
                "version_before": str(production_version.version),
                "version_after": str(production_version.version),
                "candidate_name": production_version.tags.get("candidate_name"),
            },
            "retraining_skipped": True,
        }
        _write_json(summary_path, summary)
        summary["summary_path"] = str(summary_path.relative_to(repo_root))
        return summary

    batch_payload = _load_retraining_batch(
        resolved_batch_path,
        use_clean_labels_for_evaluation=use_clean_labels_for_evaluation,
    )
    adaptation_split = _split_recent_batch_for_adaptation(
        images=batch_payload["images"],
        train_labels=batch_payload["train_labels"],
        evaluation_labels=batch_payload["evaluation_labels"],
        adaptation_test_size=resolved_adaptation_test_size,
        seed=base_seed,
    )
    fit_corpus_payload = _combine_available_training_data(
        current_training_reference=current_training_reference,
        batch_payload={
            "images": adaptation_split["train_images"],
            "train_labels": adaptation_split["train_labels"],
        },
    )
    training_split = _split_training_corpus(
        images=fit_corpus_payload["images"],
        labels=fit_corpus_payload["labels"],
        validation_size=resolved_validation_size,
        seed=base_seed,
    )
    absorbed_training_payload = _combine_available_training_data(
        current_training_reference=current_training_reference,
        batch_payload=batch_payload,
    )
    benchmark_batch = _load_labeled_batch(repo_root / "data" / "processed" / f"{BENCHMARK_BATCH_NAME}.npz")

    production_reference_model = mlflow.tensorflow.load_model(production_uri)
    production_adaptation_metrics = _evaluate_model_on_arrays(
        model=production_reference_model,
        images=adaptation_split["adaptation_images"],
        labels=adaptation_split["adaptation_labels"],
    )
    production_benchmark_metrics = _evaluate_model_on_arrays(
        model=production_reference_model,
        images=benchmark_batch["images"],
        labels=benchmark_batch["labels"],
    )

    challenger_model, challenger_history = _fit_model_from_base_uri(
        base_model_uri=production_uri,
        train_images=training_split["train_images"],
        train_labels=training_split["train_labels"],
        validation_images=training_split["validation_images"],
        validation_labels=training_split["validation_labels"],
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=base_seed,
    )
    challenger_adaptation_metrics = _evaluate_model_on_arrays(
        model=challenger_model,
        images=adaptation_split["adaptation_images"],
        labels=adaptation_split["adaptation_labels"],
    )
    challenger_benchmark_metrics = _evaluate_model_on_arrays(
        model=challenger_model,
        images=benchmark_batch["images"],
        labels=benchmark_batch["labels"],
    )

    decision = decide_challenger_promotion(
        production_adaptation_metrics=production_adaptation_metrics,
        challenger_adaptation_metrics=challenger_adaptation_metrics,
        production_benchmark_metrics=production_benchmark_metrics,
        challenger_benchmark_metrics=challenger_benchmark_metrics,
        min_adaptation_accuracy_gain=acceptance_policy["min_adaptation_accuracy_gain"],
        max_benchmark_accuracy_regression=acceptance_policy["max_benchmark_accuracy_regression"],
        keep_production_on_failure=acceptance_policy["keep_production_on_failure"],
    )

    history_path = session_dir / "history.json"
    comparison_path = session_dir / "comparison.json"
    model_path = models_root / f"{production_version.tags.get('candidate_name', 'challenger')}.keras"
    challenger_model.save(model_path)
    _write_json(history_path, challenger_history)

    promotion_payload: dict[str, Any] | None = None
    promoted_training_reference: dict[str, Any] | None = None
    version_after = str(production_version.version)
    run_name = f"retrain-{resolved_batch_name}-{session_dir.name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "registered_model_name": resolved_model_name,
                "registered_alias": alias,
                "production_version_before": str(production_version.version),
                "production_candidate_name": production_version.tags.get("candidate_name", "unknown"),
                "monitoring_summary": monitoring_summary["summary_path"],
                "retraining_batch_name": resolved_batch_name,
                "retraining_batch_path": str(resolved_batch_path.relative_to(repo_root)),
                "training_label_key": batch_payload["training_label_key"],
                "evaluation_label_key": batch_payload["evaluation_label_key"],
                "current_training_reference_batch": current_training_reference["batch_name"],
                "current_training_reference_examples": current_training_reference["num_examples"],
                "recent_labeled_examples": len(batch_payload["images"]),
                "recent_training_pool_examples": len(adaptation_split["train_labels"]),
                "adaptation_test_examples": len(adaptation_split["adaptation_labels"]),
                "fit_corpus_examples": len(fit_corpus_payload["labels"]),
                "training_examples": len(training_split["train_labels"]),
                "validation_examples": len(training_split["validation_labels"]),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_size": _serialize_split_policy(resolved_validation_size),
                "adaptation_test_size": _serialize_split_policy(resolved_adaptation_test_size),
                "force_retrain": int(force_retrain),
                "promote_on_accept": int(promote_on_accept),
            }
        )
        mlflow.log_metrics(
            {
                "production_adaptation_accuracy": production_adaptation_metrics["accuracy"],
                "production_adaptation_macro_f1": production_adaptation_metrics["macro_f1"],
                "production_adaptation_mean_confidence": production_adaptation_metrics["mean_confidence"],
                "challenger_adaptation_accuracy": challenger_adaptation_metrics["accuracy"],
                "challenger_adaptation_macro_f1": challenger_adaptation_metrics["macro_f1"],
                "challenger_adaptation_mean_confidence": challenger_adaptation_metrics["mean_confidence"],
                "production_benchmark_accuracy": production_benchmark_metrics["accuracy"],
                "production_benchmark_macro_f1": production_benchmark_metrics["macro_f1"],
                "production_benchmark_mean_confidence": production_benchmark_metrics["mean_confidence"],
                "challenger_benchmark_accuracy": challenger_benchmark_metrics["accuracy"],
                "challenger_benchmark_macro_f1": challenger_benchmark_metrics["macro_f1"],
                "challenger_benchmark_mean_confidence": challenger_benchmark_metrics["mean_confidence"],
                "adaptation_accuracy_gain": decision["adaptation_accuracy_gain"],
                "benchmark_accuracy_change": decision["benchmark_accuracy_change"],
                "benchmark_accuracy_regression": decision["benchmark_accuracy_regression"],
                "accepted": float(decision["accepted"]),
            }
        )
        mlflow.log_artifact(str(history_path), artifact_path="reports")
        mlflow.log_artifact(str(model_path), artifact_path="challenger")

        if decision["accepted"] and promote_on_accept:
            promoted_training_split = _split_training_corpus(
                images=absorbed_training_payload["images"],
                labels=absorbed_training_payload["labels"],
                validation_size=resolved_validation_size,
                seed=base_seed,
            )
            promoted_model, promoted_history = _fit_model_from_base_uri(
                base_model_uri=production_uri,
                train_images=promoted_training_split["train_images"],
                train_labels=promoted_training_split["train_labels"],
                validation_images=promoted_training_split["validation_images"],
                validation_labels=promoted_training_split["validation_labels"],
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                seed=base_seed,
            )
            promoted_training_reference = materialize_training_reference_data(
                repo_root=repo_root,
                images=absorbed_training_payload["images"],
                labels=absorbed_training_payload["labels"],
                output_path=TRAINING_REFERENCE_OUTPUT_DIR / f"production_training_reference_{session_dir.name}.npz",
                batch_name=f"production_training_reference_{session_dir.name}",
            )
            promoted_training_reference["dvc"] = _version_training_reference_with_dvc(
                repo_root=repo_root,
                batch_path=_resolve_path(repo_root, promoted_training_reference["batch_path"]),
            )
            promoted_model_path = models_root / f"{production_version.tags.get('candidate_name', 'challenger')}_promoted.keras"
            promoted_history_path = session_dir / "promotion_refit_history.json"
            promoted_model.save(promoted_model_path)
            _write_json(promoted_history_path, promoted_history)
            mlflow.log_artifact(str(promoted_history_path), artifact_path="promotion")
            mlflow.log_artifact(str(promoted_model_path), artifact_path="promoted_model")
            mlflow.log_artifact(
                str(_resolve_path(repo_root, promoted_training_reference["batch_path"])),
                artifact_path="training_reference",
            )
            mlflow.tensorflow.log_model(promoted_model, artifact_path="model")
            promotion_payload = _register_challenger_model(
                client=client,
                run_id=run.info.run_id,
                registered_model_name=resolved_model_name,
                alias=alias,
                candidate_name=production_version.tags.get("candidate_name", "unknown"),
                retraining_batch_name=resolved_batch_name,
                monitoring_summary=monitoring_summary["summary_path"],
                training_reference=promoted_training_reference,
            )
            promotion_payload["refit_history_path"] = str(promoted_history_path.relative_to(repo_root))
            promotion_payload["promoted_model_path"] = str(promoted_model_path.relative_to(repo_root))
            promotion_payload["refit_on_absorbed_training_reference"] = True
            promotion_payload["training_reference_dvc"] = promoted_training_reference["dvc"]
            version_after = str(promotion_payload["registered_version"])
        else:
            mlflow.tensorflow.log_model(challenger_model, artifact_path="model")

    summary = {
        "decision": decision,
        "monitoring_summary": monitoring_summary["summary_path"],
        "retraining_source": {
            "batch_name": resolved_batch_name,
            "batch_path": str(resolved_batch_path.relative_to(repo_root)),
            "training_label_key": batch_payload["training_label_key"],
            "evaluation_label_key": batch_payload["evaluation_label_key"],
        },
        "validation_policy": {
            "validation_size": _serialize_split_policy(resolved_validation_size),
            "seed": base_seed,
        },
        "acceptance_policy": {
            **acceptance_policy,
            "adaptation_test_size": _serialize_split_policy(resolved_adaptation_test_size),
            "benchmark_batch_name": BENCHMARK_BATCH_NAME,
        },
        "split_sizes": {
            "current_training_reference": current_training_reference["num_examples"],
            "recent_labeled": len(batch_payload["images"]),
            "recent_training_pool": len(adaptation_split["train_labels"]),
            "recent_adaptation_test": len(adaptation_split["adaptation_labels"]),
            "fit_corpus": len(fit_corpus_payload["labels"]),
            "training": len(training_split["train_labels"]),
            "validation": len(training_split["validation_labels"]),
            "absorbed_training_reference": len(absorbed_training_payload["labels"]),
        },
        "training_reference": {
            "batch_name_before": current_training_reference["batch_name"],
            "batch_path_before": current_training_reference["batch_path"],
            "num_examples_before": current_training_reference["num_examples"],
            "recent_examples_added": len(batch_payload["images"]),
            "recent_train_examples": len(adaptation_split["train_labels"]),
            "recent_adaptation_test_examples": len(adaptation_split["adaptation_labels"]),
            "batch_name_after": (
                promoted_training_reference["batch_name"]
                if promoted_training_reference is not None
                else current_training_reference["batch_name"]
            ),
            "batch_path_after": (
                promoted_training_reference["batch_path"]
                if promoted_training_reference is not None
                else current_training_reference["batch_path"]
            ),
            "num_examples_after": (
                promoted_training_reference["num_examples"]
                if promoted_training_reference is not None
                else current_training_reference["num_examples"]
            ),
            "dvc": promoted_training_reference.get("dvc") if promoted_training_reference is not None else None,
        },
        "production_model": {
            "registered_model_name": resolved_model_name,
            "alias": alias,
            "version_before": str(production_version.version),
            "version_after": version_after,
            "candidate_name": production_version.tags.get("candidate_name"),
            "source_training_run_id": production_version.tags.get("source_training_run_id"),
        },
        "production_adaptation_metrics": production_adaptation_metrics,
        "challenger_adaptation_metrics": challenger_adaptation_metrics,
        "production_benchmark_metrics": production_benchmark_metrics,
        "challenger_benchmark_metrics": challenger_benchmark_metrics,
        "production_holdout_metrics": production_adaptation_metrics,
        "challenger_holdout_metrics": challenger_adaptation_metrics,
        "challenger_model_path": str(model_path.relative_to(repo_root)),
        "history_path": str(history_path.relative_to(repo_root)),
        "comparison_path": str(comparison_path.relative_to(repo_root)),
        "promotion": promotion_payload,
    }
    _write_json(comparison_path, {
        "production": {
            "adaptation_test": production_adaptation_metrics,
            "benchmark_test": production_benchmark_metrics,
        },
        "challenger": {
            "adaptation_test": challenger_adaptation_metrics,
            "benchmark_test": challenger_benchmark_metrics,
        },
        "decision": decision,
    })
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(repo_root))
    return summary


def decide_challenger_promotion(
    production_adaptation_metrics: dict[str, float],
    challenger_adaptation_metrics: dict[str, float],
    production_benchmark_metrics: dict[str, float],
    challenger_benchmark_metrics: dict[str, float],
    min_adaptation_accuracy_gain: float,
    max_benchmark_accuracy_regression: float,
    keep_production_on_failure: bool,
) -> dict[str, Any]:
    adaptation_accuracy_gain = float(
        challenger_adaptation_metrics["accuracy"] - production_adaptation_metrics["accuracy"]
    )
    benchmark_accuracy_change = float(
        challenger_benchmark_metrics["accuracy"] - production_benchmark_metrics["accuracy"]
    )
    benchmark_accuracy_regression = float(
        production_benchmark_metrics["accuracy"] - challenger_benchmark_metrics["accuracy"]
    )
    adaptation_gate_met = adaptation_accuracy_gain >= float(min_adaptation_accuracy_gain)
    benchmark_gate_met = benchmark_accuracy_regression <= float(max_benchmark_accuracy_regression)
    accepted = adaptation_gate_met and benchmark_gate_met
    action = "promote_challenger" if accepted else "keep_production"
    if not keep_production_on_failure and not accepted:
        action = "manual_review"

    if accepted:
        reason = "adaptation_gain_met_and_benchmark_held"
    elif not adaptation_gate_met and not benchmark_gate_met:
        reason = "adaptation_gain_below_threshold_and_benchmark_regression_exceeded"
    elif not adaptation_gate_met:
        reason = "adaptation_gain_below_threshold"
    else:
        reason = "benchmark_regression_exceeded"

    return {
        "accepted": accepted,
        "action": action,
        "reason": reason,
        "adaptation_accuracy_gain": adaptation_accuracy_gain,
        "required_adaptation_accuracy_gain": float(min_adaptation_accuracy_gain),
        "benchmark_accuracy_change": benchmark_accuracy_change,
        "benchmark_accuracy_regression": benchmark_accuracy_regression,
        "max_allowed_benchmark_accuracy_regression": float(max_benchmark_accuracy_regression),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the retraining decision workflow for one recent labeled batch.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument(
        "--monitoring-summary-path",
        default=None,
        help="Optional monitoring summary JSON. If omitted, a monitoring run is created first.",
    )
    parser.add_argument(
        "--monitoring-batch-name",
        default=None,
        help="Batch used for the monitoring gate when no monitoring summary is provided.",
    )
    parser.add_argument(
        "--retraining-batch-name",
        default=None,
        help="Named retraining batch under data/processed.",
    )
    parser.add_argument("--retraining-batch-path", default=None, help="Optional explicit retraining batch .npz path.")
    parser.add_argument("--output-dir", default="artifacts/retraining", help="Folder for retraining reports.")
    parser.add_argument("--models-dir", default="models/retraining", help="Folder for saved challenger models.")
    parser.add_argument("--registered-model-name", default=None, help="Optional MLflow model name override.")
    parser.add_argument("--alias", default=MODEL_ALIAS, help="Model alias to compare and optionally update.")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate for fine-tuning.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Optional validation policy override. Fractions < 1 are treated as percentages of the fit corpus.",
    )
    parser.add_argument(
        "--adaptation-test-size",
        type=float,
        default=None,
        help="Optional adaptation-test override. Fractions < 1 are treated as percentages of the recent labeled batch.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help="Deprecated alias for --adaptation-test-size.",
    )
    parser.add_argument(
        "--disable-clean-labels",
        action="store_true",
        help="Always evaluate with labels instead of clean_labels.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Bypass the monitoring gate and retrain anyway.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Run the challenger comparison without moving the production alias.",
    )
    args = parser.parse_args(argv)

    summary = run_retraining_workflow(
        params_path=args.params,
        monitoring_summary_path=args.monitoring_summary_path,
        monitoring_batch_name=args.monitoring_batch_name,
        retraining_batch_name=args.retraining_batch_name,
        retraining_batch_path=args.retraining_batch_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        adaptation_test_size=args.adaptation_test_size,
        holdout_fraction=args.holdout_fraction,
        use_clean_labels_for_evaluation=not args.disable_clean_labels,
        force_retrain=args.force_retrain,
        promote_on_accept=not args.no_promote,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _resolve_monitoring_summary(
    repo_root: Path,
    params_path: Path,
    monitoring_summary_path: str | Path | None,
    monitoring_batch_name: str | None,
    retraining_batch_name: str | None,
    retraining_batch_path: str | Path | None,
) -> dict[str, Any]:
    if monitoring_summary_path is not None:
        path = _resolve_path(repo_root, monitoring_summary_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(path.relative_to(repo_root))
        return payload

    batch_name = monitoring_batch_name or retraining_batch_name
    return analyze_batch_monitoring(
        params_path=params_path,
        batch_name=batch_name,
        batch_path=retraining_batch_path if batch_name is None else None,
    )


def _resolve_retraining_source(
    repo_root: Path,
    monitoring_summary: dict[str, Any],
    retraining_batch_name: str | None,
    retraining_batch_path: str | Path | None,
) -> tuple[str, Path]:
    if retraining_batch_path is not None:
        resolved_path = _resolve_path(repo_root, retraining_batch_path)
        return resolved_path.stem, resolved_path

    if retraining_batch_name is not None:
        return retraining_batch_name, (repo_root / "data" / "processed" / f"{retraining_batch_name}.npz").resolve()

    return monitoring_summary["batch_name"], _resolve_path(repo_root, monitoring_summary["batch_path"])


def _load_retraining_batch(
    batch_path: Path,
    use_clean_labels_for_evaluation: bool,
) -> dict[str, Any]:
    with np.load(batch_path) as batch:
        images = batch["images"].astype(np.float32)
        train_labels = batch["labels"].astype(np.int64)
        evaluation_label_key = "clean_labels" if use_clean_labels_for_evaluation and "clean_labels" in batch.files else "labels"
        evaluation_labels = batch[evaluation_label_key].astype(np.int64)

    return {
        "images": images,
        "train_labels": train_labels,
        "evaluation_labels": evaluation_labels,
        "training_label_key": "labels",
        "evaluation_label_key": evaluation_label_key,
    }


def _split_recent_batch_for_adaptation(
    images: np.ndarray,
    train_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    adaptation_test_size: float | int,
    seed: int,
) -> dict[str, np.ndarray]:
    if len(images) < 2:
        raise ValueError("Retraining batch must contain at least 2 samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    adaptation_size = _resolve_split_size(
        adaptation_test_size,
        reference_total=len(images),
        split_name="adaptation_test_size",
        minimum=1,
        reserve_examples=1,
    )
    adaptation_indices = indices[-adaptation_size:]
    train_indices = indices[:-adaptation_size]

    return {
        "train_images": images[train_indices],
        "train_labels": train_labels[train_indices],
        "adaptation_images": images[adaptation_indices],
        "adaptation_labels": evaluation_labels[adaptation_indices],
    }


def _split_training_corpus(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: float | int,
    seed: int,
) -> dict[str, np.ndarray]:
    if len(images) < 2:
        raise ValueError("Training corpus must contain at least 2 samples")

    validation_examples = _resolve_split_size(
        validation_size,
        reference_total=len(images),
        split_name="validation_size",
        minimum=1,
        reserve_examples=1,
    )
    indices = np.random.default_rng(seed).permutation(len(images))
    validation_indices = indices[-validation_examples:]
    train_indices = indices[:-validation_examples]

    return {
        "train_images": images[train_indices],
        "train_labels": labels[train_indices],
        "validation_images": images[validation_indices],
        "validation_labels": labels[validation_indices],
    }


def _combine_available_training_data(
    current_training_reference: dict[str, Any],
    batch_payload: dict[str, Any],
) -> dict[str, np.ndarray]:
    recent_images = np.asarray(batch_payload["images"], dtype=np.float32)
    recent_labels = np.asarray(batch_payload["train_labels"], dtype=np.int64)
    if len(recent_images) != len(recent_labels):
        raise ValueError("Recent retraining images and labels must have matching lengths")

    reference_images = np.asarray(current_training_reference["images"], dtype=np.float32)
    reference_labels = np.asarray(current_training_reference["labels"], dtype=np.int64)
    if len(reference_images) != len(reference_labels):
        raise ValueError("Current training reference images and labels must have matching lengths")

    return {
        "images": np.concatenate([reference_images, recent_images], axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate([reference_labels, recent_labels], axis=0).astype(np.int64, copy=False),
    }


def _resolve_acceptance_policy(params: dict[str, Any]) -> dict[str, Any]:
    acceptance_config = params["retraining"]["acceptance"]
    min_adaptation_accuracy_gain = float(
        acceptance_config.get("min_adaptation_accuracy_gain", acceptance_config.get("min_accuracy_gain", 0.0))
    )
    max_benchmark_accuracy_regression = float(acceptance_config.get("max_benchmark_accuracy_regression", 0.0))

    return {
        "min_adaptation_accuracy_gain": min_adaptation_accuracy_gain,
        "max_benchmark_accuracy_regression": max_benchmark_accuracy_regression,
        "keep_production_on_failure": bool(acceptance_config["keep_production_on_failure"]),
    }


def _resolve_split_size(
    raw_size: float | int,
    reference_total: int,
    split_name: str,
    minimum: int = 1,
    reserve_examples: int = 0,
) -> int:
    if reference_total <= reserve_examples:
        raise ValueError(f"Not enough samples to reserve {reserve_examples} examples outside {split_name}")

    numeric_size = float(raw_size)
    if numeric_size <= 0:
        raise ValueError(f"{split_name} must be positive")

    if numeric_size < 1.0:
        resolved_size = int(round(reference_total * numeric_size))
    else:
        resolved_size = int(numeric_size)

    maximum = reference_total - reserve_examples
    resolved_size = max(minimum, resolved_size)
    resolved_size = min(resolved_size, maximum)
    if resolved_size <= 0:
        raise ValueError(f"{split_name} resolved to zero examples")
    return resolved_size


def _serialize_split_policy(raw_size: float | int) -> float | int:
    numeric_size = float(raw_size)
    return int(numeric_size) if numeric_size >= 1.0 and numeric_size.is_integer() else numeric_size


def _load_labeled_batch(batch_path: Path) -> dict[str, np.ndarray]:
    with np.load(batch_path) as batch:
        return {
            "images": batch["images"].astype(np.float32),
            "labels": batch["labels"].astype(np.int64),
        }


def _fit_model_from_base_uri(
    base_model_uri: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[keras.Model, dict[str, list[float]]]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    model = mlflow.tensorflow.load_model(base_model_uri)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model, history.history


def _version_training_reference_with_dvc(repo_root: Path, batch_path: Path) -> dict[str, Any]:
    dvc_executable = shutil.which("dvc")
    if dvc_executable is None:
        local_dvc = repo_root / ".venv" / "bin" / "dvc"
        if local_dvc.exists():
            dvc_executable = local_dvc.as_posix()

    if dvc_executable is None:
        raise FileNotFoundError("Could not find a DVC executable to version the promoted training reference")

    relative_batch_path = str(batch_path.relative_to(repo_root))
    result = subprocess.run(
        [dvc_executable, "add", relative_batch_path],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_output = result.stderr.strip() or result.stdout.strip() or "unknown dvc add error"
        raise RuntimeError(f"DVC failed to add {relative_batch_path}: {error_output}")

    dvc_file_path = batch_path.with_suffix(batch_path.suffix + ".dvc")
    return {
        "tracked": dvc_file_path.exists(),
        "batch_path": relative_batch_path,
        "dvc_file_path": str(dvc_file_path.relative_to(repo_root)) if dvc_file_path.exists() else None,
        "command": f"{Path(dvc_executable).name} add {relative_batch_path}",
    }


def _evaluate_model_on_arrays(model: Any, images: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probabilities = model.predict(images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "macro_f1": float(f1_score(labels, predicted_labels, average="macro")),
        "mean_confidence": float(np.mean(confidences)),
        "num_examples": int(len(labels)),
        "predicted_distribution": np.bincount(predicted_labels, minlength=10).tolist(),
    }


def _register_challenger_model(
    client: MlflowClient,
    run_id: str,
    registered_model_name: str,
    alias: str,
    candidate_name: str,
    retraining_batch_name: str,
    monitoring_summary: str,
    training_reference: dict[str, Any],
) -> dict[str, Any]:
    registered_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=registered_model_name,
    )
    _wait_for_model_version(client, registered_model_name, registered_version.version)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=registered_version.version,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key="source_training_run_id",
        value=run_id,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key="candidate_name",
        value=candidate_name,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key="retraining_batch_name",
        value=retraining_batch_name,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key="monitoring_summary",
        value=monitoring_summary,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key=TRAINING_REFERENCE_NAME_TAG,
        value=str(training_reference["batch_name"]),
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key=TRAINING_REFERENCE_PATH_TAG,
        value=str(training_reference["batch_path"]),
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key=TRAINING_REFERENCE_SIZE_TAG,
        value=str(training_reference["num_examples"]),
    )

    return {
        "registered_model_name": registered_model_name,
        "registered_version": str(registered_version.version),
        "alias": alias,
        "run_id": run_id,
        "training_reference": training_reference,
    }


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


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    return load_runtime_config(repo_root)


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    return resolve_repo_bound_path(repo_root, raw_path)


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    return resolve_tracking_uri(repo_root, raw_tracking_uri)


def _resolve_artifact_root(repo_root: Path, raw_artifact_root: str) -> Path:
    return resolve_artifact_root(repo_root, {"MLFLOW_ARTIFACT_ROOT": raw_artifact_root})


def _ensure_experiment(experiment_name: str, artifact_root: Path) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())