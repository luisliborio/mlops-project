from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mlops_mnist.utils.mlflow_compat import (
    resolve_repo_bound_path,
    resolve_runtime_write_path,
    serialize_repo_bound_path,
)


LIFECYCLE_STATE_RELATIVE_PATH = Path("artifacts/ui_lifecycle/active_session.json")


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def create_lifecycle_state(
    batch_name: str,
    batch_size: int,
    baseline_snapshot_path: str | None = None,
    baseline_reference: dict[str, Any] | None = None,
    record_mode: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "active_batch_name": batch_name,
        "batch_size": int(batch_size),
        "cursor": 0,
        "step_counter": 0,
        "stream_exhausted": False,
        "record_mode": bool(record_mode),
        "baseline_snapshot_path": baseline_snapshot_path,
        "baseline_reference": baseline_reference,
        "latest_preview": None,
        "recorded_batches": [],
        "event_log": [],
        "unsupervised_history": [],
        "supervised_history": [],
        "retraining_history": [],
        "training_reference_history": [],
        "current_alerts": [],
        "latest_monitoring_summary_path": None,
        "latest_retraining_summary_path": None,
    }


def load_lifecycle_state(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    state_path = lifecycle_state_path(repo_root)
    if not state_path.exists():
        return create_lifecycle_state(batch_name="", batch_size=16)
    return _normalize_lifecycle_state_paths(
        repo_root,
        json.loads(state_path.read_text(encoding="utf-8")),
    )


def save_lifecycle_state(repo_root: str | Path, state: dict[str, Any]) -> Path:
    repo_root = Path(repo_root)
    state_path = lifecycle_state_path(repo_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


def reset_lifecycle_state(
    repo_root: str | Path,
    batch_name: str,
    batch_size: int,
    baseline_snapshot_path: str | None = None,
    baseline_reference: dict[str, Any] | None = None,
    record_mode: bool = False,
) -> dict[str, Any]:
    state = create_lifecycle_state(
        batch_name=batch_name,
        batch_size=batch_size,
        baseline_snapshot_path=baseline_snapshot_path,
        baseline_reference=baseline_reference,
        record_mode=record_mode,
    )
    append_event_log(
        state,
        level="info",
        message=f"Scenario reset for {batch_name} with batch size {batch_size}.",
    )
    save_lifecycle_state(repo_root, state)
    return state


def switch_lifecycle_batch(
    repo_root: str | Path,
    state: dict[str, Any],
    batch_name: str,
    batch_size: int,
) -> dict[str, Any]:
    previous_batch_name = state.get("active_batch_name") or "none"
    state["active_batch_name"] = batch_name
    state["batch_size"] = int(batch_size)
    state["cursor"] = 0
    state["stream_exhausted"] = False
    state["latest_preview"] = None
    state["current_alerts"] = []
    append_event_log(
        state,
        level="info",
        message=(
            f"Incoming source batch switched from {previous_batch_name} to {batch_name}. "
            "The stream cursor restarted and lifecycle history was preserved."
        ),
    )
    save_lifecycle_state(repo_root, state)
    return state


def lifecycle_state_path(repo_root: str | Path) -> Path:
    return resolve_runtime_write_path(Path(repo_root), LIFECYCLE_STATE_RELATIVE_PATH)


def compute_cursor_slice(cursor: int, batch_size: int, total_examples: int) -> dict[str, Any]:
    if total_examples < 0:
        raise ValueError("total_examples must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if cursor >= total_examples:
        return {
            "indices": [],
            "start_index": int(cursor),
            "end_index": int(cursor),
            "next_cursor": int(cursor),
            "stream_exhausted": True,
        }

    end_index = min(cursor + batch_size, total_examples)
    return {
        "indices": list(range(cursor, end_index)),
        "start_index": int(cursor),
        "end_index": int(end_index),
        "next_cursor": int(end_index),
        "stream_exhausted": end_index >= total_examples,
    }


def advance_stream_cursor(state: dict[str, Any], total_examples: int, batch_size: int) -> dict[str, Any]:
    slice_payload = compute_cursor_slice(
        cursor=int(state.get("cursor", 0)),
        batch_size=int(batch_size),
        total_examples=int(total_examples),
    )

    if slice_payload["indices"]:
        state["cursor"] = int(slice_payload["next_cursor"])
        state["batch_size"] = int(batch_size)
        state["stream_exhausted"] = bool(slice_payload["stream_exhausted"])
        state["step_counter"] = int(state.get("step_counter", 0)) + 1
        slice_payload["step"] = int(state["step_counter"])
    else:
        state["stream_exhausted"] = True
        slice_payload["step"] = int(state.get("step_counter", 0))

    return slice_payload


def append_event_log(state: dict[str, Any], level: str, message: str) -> None:
    state.setdefault("event_log", []).append(
        {
            "timestamp": _utc_now(),
            "level": level,
            "message": message,
        }
    )


def record_preview_batch(state: dict[str, Any], preview_payload: dict[str, Any]) -> None:
    state["latest_preview"] = preview_payload


def append_recorded_batch(
    state: dict[str, Any],
    batch_name: str,
    step: int,
    slice_payload: dict[str, Any],
    prediction: dict[str, Any],
    latency_ms: float,
    histogram_js_divergence: float,
    confidence_drop: float,
    alerts: list[str],
    clean_labels_available: bool,
) -> dict[str, Any]:
    record = {
        "step": int(step),
        "batch_name": batch_name,
        "start_index": int(slice_payload["start_index"]),
        "end_index": int(slice_payload["end_index"]),
        "indices": [int(index) for index in slice_payload["indices"]],
        "labeled": False,
        "model_version": prediction.get("model_version"),
        "mean_confidence": float(prediction.get("mean_confidence", 0.0)),
        "latency_ms": float(latency_ms),
        "histogram_js_divergence": float(histogram_js_divergence),
        "confidence_drop": float(confidence_drop),
        "alerts": list(alerts),
        "clean_labels_available": bool(clean_labels_available),
        "predicted_labels": [
            int(row["predicted_label"]) for row in prediction.get("predictions", [])
        ],
        "confidences": [
            float(row["confidence"]) for row in prediction.get("predictions", [])
        ],
    }
    state.setdefault("recorded_batches", []).append(record)
    state.setdefault("unsupervised_history", []).append(
        {
            "step": int(step),
            "recorded_examples": count_recorded_samples(state),
            "mean_confidence": float(record["mean_confidence"]),
            "latency_ms": float(latency_ms),
            "histogram_js_divergence": float(histogram_js_divergence),
            "confidence_drop": float(confidence_drop),
        }
    )
    state["current_alerts"] = list(alerts)
    return record


def count_recorded_samples(state: dict[str, Any]) -> int:
    return int(
        sum(len(record.get("indices", [])) for record in state.get("recorded_batches", []))
    )


def count_pending_labeled_samples(state: dict[str, Any]) -> int:
    return int(
        sum(
            len(record.get("indices", []))
            for record in state.get("recorded_batches", [])
            if not record.get("labeled", False)
        )
    )


def count_ready_labeled_samples(state: dict[str, Any]) -> int:
    return int(
        sum(
            len(record.get("indices", []))
            for record in state.get("recorded_batches", [])
            if record.get("labeled", False)
        )
    )


def label_pending_batches(state: dict[str, Any]) -> int:
    labeled_now = 0
    for record in state.get("recorded_batches", []):
        if record.get("labeled", False):
            continue
        record["labeled"] = True
        labeled_now += len(record.get("indices", []))
    return int(labeled_now)


def clear_labeled_batches(state: dict[str, Any]) -> int:
    labeled_records = [
        record for record in state.get("recorded_batches", []) if record.get("labeled", False)
    ]
    cleared_samples = int(sum(len(record.get("indices", [])) for record in labeled_records))
    if cleared_samples <= 0:
        return 0

    state["recorded_batches"] = [
        record for record in state.get("recorded_batches", []) if not record.get("labeled", False)
    ]
    state["supervised_history"] = []
    state["current_alerts"] = []
    state["latest_monitoring_summary_path"] = None
    state["latest_retraining_summary_path"] = None
    return cleared_samples


def latest_recorded_distribution(state: dict[str, Any]) -> list[int]:
    records = state.get("recorded_batches", [])
    if not records:
        return [0] * 10
    predicted_labels = np.asarray(records[-1].get("predicted_labels", []), dtype=np.int64)
    return np.bincount(predicted_labels, minlength=10).tolist()


def materialize_recorded_batch(
    repo_root: str | Path,
    state: dict[str, Any],
    output_path: str | Path,
    labeled_only: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    records = [
        record
        for record in state.get("recorded_batches", [])
        if not labeled_only or record.get("labeled", False)
    ]
    if not records:
        raise ValueError("No recorded batches available for lifecycle materialization")

    source_batches = list(dict.fromkeys(str(record.get("batch_name", "")) for record in records))
    batch_cache: dict[str, dict[str, np.ndarray]] = {}
    image_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    clean_label_chunks: list[np.ndarray] = []
    has_clean_labels = False

    for record in records:
        record_batch_name = str(record.get("batch_name") or state.get("active_batch_name") or "")
        if not record_batch_name:
            raise ValueError("Recorded lifecycle batch is missing its source batch name")

        if record_batch_name not in batch_cache:
            source_batch_path = repo_root / "data" / "processed" / f"{record_batch_name}.npz"
            if not source_batch_path.exists():
                raise FileNotFoundError(f"Missing source batch file: {source_batch_path}")

            with np.load(source_batch_path) as batch:
                batch_payload: dict[str, np.ndarray] = {
                    "images": batch["images"].astype(np.float32),
                    "labels": batch["labels"].astype(np.int64),
                }
                if "clean_labels" in batch.files:
                    batch_payload["clean_labels"] = batch["clean_labels"].astype(np.int64)
                batch_cache[record_batch_name] = batch_payload

        indices = [int(index) for index in record.get("indices", [])]
        if not indices:
            continue

        batch_payload = batch_cache[record_batch_name]
        image_chunks.append(batch_payload["images"][indices])
        label_chunks.append(batch_payload["labels"][indices])
        if "clean_labels" in batch_payload:
            clean_label_chunks.append(batch_payload["clean_labels"][indices])
            has_clean_labels = True
        else:
            clean_label_chunks.append(batch_payload["labels"][indices])

    if not image_chunks:
        raise ValueError("No recorded batches available for lifecycle materialization")

    images = np.concatenate(image_chunks, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int64, copy=False)
    clean_labels = None
    if has_clean_labels:
        clean_labels = np.concatenate(clean_label_chunks, axis=0).astype(np.int64, copy=False)

    output_path = resolve_runtime_write_path(repo_root, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clean_labels is None:
        np.savez_compressed(output_path, images=images, labels=labels)
    else:
        np.savez_compressed(output_path, images=images, labels=labels, clean_labels=clean_labels)

    return {
        "batch_name": "ui_lifecycle_recorded_buffer",
        "batch_path": str(output_path),
        "num_examples": int(images.shape[0]),
        "has_clean_labels": clean_labels is not None,
        "source_batches": source_batches,
    }


def ensure_baseline_reference(
    repo_root: str | Path,
    params_path: str | Path = "params.yaml",
    baseline_snapshot_path: str | Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    snapshot_payload: dict[str, Any]

    if baseline_snapshot_path is not None and not force_refresh:
        resolved_path = _resolve_path(repo_root, baseline_snapshot_path)
        if resolved_path.exists():
            snapshot_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
            snapshot_payload["summary_path"] = str(resolved_path.relative_to(repo_root))
            return _extract_baseline_reference(repo_root, snapshot_payload)

    from mlops_mnist.training.evaluation import evaluate_registered_model

    snapshot_payload = evaluate_registered_model(
        params_path=_resolve_path(repo_root, params_path),
        output_dir="artifacts/ui_lifecycle/baseline",
        freeze_baseline=True,
    )
    return _extract_baseline_reference(repo_root, snapshot_payload)


def compute_js_divergence(reference_values: np.ndarray, current_values: np.ndarray) -> float:
    reference = np.asarray(reference_values, dtype=np.float64)
    current = np.asarray(current_values, dtype=np.float64)
    reference = reference / np.clip(reference.sum(), 1e-12, None)
    current = current / np.clip(current.sum(), 1e-12, None)
    reference = np.clip(reference, 1e-12, None)
    current = np.clip(current, 1e-12, None)
    midpoint = 0.5 * (reference + current)
    return float(
        0.5
        * (
            np.sum(reference * np.log(reference / midpoint))
            + np.sum(current * np.log(current / midpoint))
        )
    )


def compute_histogram_js_divergence(reference_images: np.ndarray, current_images: np.ndarray, bins: int = 32) -> float:
    reference_hist, _ = np.histogram(reference_images.reshape(-1), bins=bins, range=(0.0, 1.0))
    current_hist, _ = np.histogram(current_images.reshape(-1), bins=bins, range=(0.0, 1.0))
    return compute_js_divergence(reference_hist, current_hist)


def summarize_unsupervised_alerts(
    thresholds: dict[str, float],
    baseline_mean_confidence: float,
    current_mean_confidence: float,
    histogram_js_divergence: float,
) -> dict[str, Any]:
    confidence_drop = float(baseline_mean_confidence - current_mean_confidence)
    alerts = []
    if confidence_drop > float(thresholds["max_confidence_drop"]):
        alerts.append("confidence_drop_exceeded")
    if histogram_js_divergence > float(thresholds["max_histogram_js_divergence"]):
        alerts.append("pixel_histogram_js_divergence_exceeded")

    return {
        "confidence_drop": confidence_drop,
        "histogram_js_divergence": float(histogram_js_divergence),
        "alerts": alerts,
    }


def run_lifecycle_monitoring(
    repo_root: str | Path,
    state: dict[str, Any],
    params_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    if count_ready_labeled_samples(state) <= 0:
        raise ValueError("No labeled lifecycle data is ready for monitoring")

    repo_root = Path(repo_root)
    baseline_reference = ensure_baseline_reference(
        repo_root=repo_root,
        params_path=params_path,
        baseline_snapshot_path=state.get("baseline_snapshot_path"),
    )
    state["baseline_snapshot_path"] = baseline_reference["summary_path"]
    state["baseline_reference"] = baseline_reference

    materialized_batch = materialize_recorded_batch(
        repo_root=repo_root,
        state=state,
        output_path=repo_root / "artifacts" / "ui_lifecycle" / "monitoring" / "current_labeled_batch.npz",
        labeled_only=True,
    )

    from mlops_mnist.monitoring import analyze_batch_monitoring

    monitoring_summary = analyze_batch_monitoring(
        params_path=_resolve_path(repo_root, params_path),
        batch_path=materialized_batch["batch_path"],
        baseline_snapshot_path=baseline_reference["summary_path"],
        use_clean_labels=False,
        output_dir="artifacts/ui_lifecycle/monitoring",
    )

    state["latest_monitoring_summary_path"] = monitoring_summary["summary_path"]
    state["current_alerts"] = list(monitoring_summary.get("alerts", []))
    state.setdefault("supervised_history", []).append(
        {
            "step": int(state.get("step_counter", 0)),
            "labeled_examples": count_ready_labeled_samples(state),
            "accuracy": float(monitoring_summary["current_batch"]["accuracy"]),
            "macro_f1": float(monitoring_summary["current_batch"]["macro_f1"]),
            "mean_confidence": float(monitoring_summary["current_batch"]["mean_confidence"]),
            "accuracy_drop": float(monitoring_summary["drift_metrics"]["accuracy_drop"]),
            "confidence_drop": float(monitoring_summary["drift_metrics"]["confidence_drop"]),
            "histogram_js_divergence": float(
                monitoring_summary["drift_metrics"]["pixel_intensity_histogram_js_divergence"]
            ),
            "alerts": list(monitoring_summary.get("alerts", [])),
            "summary_path": monitoring_summary["summary_path"],
        }
    )
    return monitoring_summary


def run_lifecycle_retraining(
    repo_root: str | Path,
    state: dict[str, Any],
    params_path: str | Path = "params.yaml",
    force_retrain: bool = False,
) -> dict[str, Any]:
    if count_ready_labeled_samples(state) <= 0:
        raise ValueError("No labeled lifecycle data is ready for retraining")

    repo_root = Path(repo_root)
    if not state.get("latest_monitoring_summary_path"):
        monitoring_summary = run_lifecycle_monitoring(repo_root=repo_root, state=state, params_path=params_path)
        monitoring_summary_path = monitoring_summary["summary_path"]
    else:
        monitoring_summary_path = str(state["latest_monitoring_summary_path"])

    materialized_batch = materialize_recorded_batch(
        repo_root=repo_root,
        state=state,
        output_path=repo_root / "artifacts" / "ui_lifecycle" / "retraining" / "current_retraining_batch.npz",
        labeled_only=True,
    )

    from mlops_mnist.training import run_retraining_workflow

    retraining_summary = run_retraining_workflow(
        params_path=_resolve_path(repo_root, params_path),
        monitoring_summary_path=monitoring_summary_path,
        retraining_batch_path=materialized_batch["batch_path"],
        use_clean_labels_for_evaluation=bool(materialized_batch["has_clean_labels"]),
        force_retrain=force_retrain,
    )

    state["latest_retraining_summary_path"] = retraining_summary["summary_path"]
    state.setdefault("retraining_history", []).append(
        {
            "step": int(state.get("step_counter", 0)),
            "accepted": bool(retraining_summary["decision"]["accepted"]),
            "action": retraining_summary["decision"]["action"],
            "reason": retraining_summary["decision"]["reason"],
            "accuracy_gain": float(retraining_summary["decision"]["accuracy_gain"]),
            "adaptation_accuracy_gain": float(retraining_summary["decision"]["adaptation_accuracy_gain"]),
            "benchmark_accuracy_regression": float(retraining_summary["decision"]["benchmark_accuracy_regression"]),
            "version_before": str(retraining_summary["production_model"]["version_before"]),
            "version_after": str(retraining_summary["production_model"]["version_after"]),
            "summary_path": retraining_summary["summary_path"],
        }
    )

    if retraining_summary.get("promotion"):
        state.setdefault("training_reference_history", []).append(
            {
                "step": int(state.get("step_counter", 0)),
                "model_version": str(retraining_summary["production_model"]["version_after"]),
                "training_reference_name": retraining_summary["training_reference"]["batch_name_after"],
                "training_reference_examples": int(retraining_summary["training_reference"]["num_examples_after"]),
                "recent_examples_absorbed": int(retraining_summary["training_reference"]["recent_examples_added"]),
                "source_batch_name": retraining_summary["retraining_source"]["batch_name"],
                "dvc_tracked": bool(retraining_summary["training_reference"].get("dvc", {}).get("tracked", False)),
                "dvc_file_path": retraining_summary["training_reference"].get("dvc", {}).get("dvc_file_path"),
                "summary_path": retraining_summary["summary_path"],
            }
        )
        baseline_reference = ensure_baseline_reference(
            repo_root=repo_root,
            params_path=params_path,
            baseline_snapshot_path=state.get("baseline_snapshot_path"),
            force_refresh=True,
        )
        state["baseline_snapshot_path"] = baseline_reference["summary_path"]
        state["baseline_reference"] = baseline_reference
        state["latest_monitoring_summary_path"] = None
        state["current_alerts"] = []

    return retraining_summary


def _extract_baseline_reference(repo_root: Path, snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    for report in snapshot_payload.get("reports", []):
        if report["batch_name"] == "baseline_test":
            reference_data = snapshot_payload.get("production_reference_data")
            if isinstance(reference_data, dict) and reference_data.get("batch_path"):
                reference_data_name = str(reference_data.get("batch_name") or Path(str(reference_data["batch_path"])).stem)
                reference_data_path = serialize_repo_bound_path(repo_root, reference_data["batch_path"])
                reference_data_examples = int(reference_data.get("num_examples", 0))
            else:
                reference_data_name = report["batch_name"]
                reference_data_path = serialize_repo_bound_path(repo_root, report["batch_path"])
                reference_data_examples = int(report.get("num_examples", 0))

            return {
                "summary_path": snapshot_payload["summary_path"],
                "batch_name": report["batch_name"],
                "batch_path": serialize_repo_bound_path(repo_root, report["batch_path"]),
                "accuracy": float(report["accuracy"]),
                "macro_f1": float(report["macro_f1"]),
                "mean_confidence": float(report["mean_confidence"]),
                "predicted_distribution": list(report["predicted_distribution"]),
                "reference_data_name": reference_data_name,
                "reference_data_path": reference_data_path,
                "reference_data_examples": reference_data_examples,
            }
    raise ValueError("baseline_test not found in baseline snapshot")


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    return resolve_repo_bound_path(repo_root, raw_path)


def _normalize_lifecycle_state_paths(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    for key in ("baseline_snapshot_path", "latest_monitoring_summary_path", "latest_retraining_summary_path"):
        if state.get(key):
            state[key] = serialize_repo_bound_path(repo_root, state[key])

    baseline_reference = state.get("baseline_reference")
    if isinstance(baseline_reference, dict):
        if baseline_reference.get("summary_path"):
            baseline_reference["summary_path"] = serialize_repo_bound_path(
                repo_root,
                baseline_reference["summary_path"],
            )
        if baseline_reference.get("batch_path"):
            baseline_reference["batch_path"] = serialize_repo_bound_path(
                repo_root,
                baseline_reference["batch_path"],
            )
        if baseline_reference.get("reference_data_path"):
            baseline_reference["reference_data_path"] = serialize_repo_bound_path(
                repo_root,
                baseline_reference["reference_data_path"],
            )

    for key in ("retraining_history", "training_reference_history"):
        for entry in state.get(key, []):
            if entry.get("summary_path"):
                entry["summary_path"] = serialize_repo_bound_path(repo_root, entry["summary_path"])
            if entry.get("dvc_file_path"):
                entry["dvc_file_path"] = serialize_repo_bound_path(repo_root, entry["dvc_file_path"])

    return state


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()