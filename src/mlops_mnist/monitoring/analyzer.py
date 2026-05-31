from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mlops_mnist.training.evaluation import evaluate_registered_model
from mlops_mnist.utils.mlflow_compat import (
    resolve_repo_bound_path,
    resolve_runtime_write_path,
    serialize_repo_bound_path,
)


BASELINE_REFERENCE_BATCH = "baseline_test"


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def analyze_batch_monitoring(
    params_path: str | Path = "params.yaml",
    batch_name: str | None = None,
    batch_path: str | Path | None = None,
    baseline_snapshot_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/monitoring",
    max_samples: int | None = None,
    use_clean_labels: bool = False,
) -> dict[str, Any]:
    # Compare one labeled batch against the baseline and decide whether to alert.
    params_path = Path(params_path).resolve()
    repo_root = params_path.parent
    params = load_project_params(params_path)
    session_dir = resolve_runtime_write_path(repo_root, output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    # Load the baseline snapshot that defines the reference behavior.
    baseline_snapshot = _resolve_baseline_snapshot(
        repo_root=repo_root,
        params_path=params_path,
        baseline_snapshot_path=baseline_snapshot_path,
    )
    baseline_report = _select_report(baseline_snapshot, BASELINE_REFERENCE_BATCH)
    histogram_reference = _select_histogram_reference(baseline_snapshot, baseline_report)

    evaluation_summary = evaluate_registered_model(
        params_path=params_path,
        batch_name=batch_name,
        batch_path=batch_path,
        output_dir=session_dir / "evaluation",
        max_samples=max_samples,
        use_clean_labels=use_clean_labels,
    )
    current_report = evaluation_summary["reports"][0]

    # Compute drift signals used by monitoring and retraining.
    thresholds = params["monitoring"]["thresholds"]
    min_recent_samples = int(params["retraining"]["min_recent_samples"])
    histogram_js_divergence = compute_pixel_histogram_js_divergence(
        repo_root=repo_root,
        baseline_batch_path=histogram_reference["batch_path"],
        current_batch_path=current_report["batch_path"],
    )
    predicted_distribution_js_divergence = compute_js_divergence(
        baseline_report["predicted_distribution"],
        current_report["predicted_distribution"],
    )
    confidence_drop = float(baseline_report["mean_confidence"] - current_report["mean_confidence"])
    accuracy_drop = float(baseline_report["accuracy"] - current_report["accuracy"])

    # Convert metric changes into alert names.
    alerts = []
    if float(current_report["accuracy"]) < float(thresholds["min_rolling_accuracy"]):
        alerts.append("rolling_accuracy_below_threshold")
    if confidence_drop > float(thresholds["max_confidence_drop"]):
        alerts.append("confidence_drop_exceeded")
    if histogram_js_divergence > float(thresholds["max_histogram_js_divergence"]):
        alerts.append("pixel_histogram_js_divergence_exceeded")

    summary = {
        "batch_name": current_report["batch_name"],
        "batch_path": serialize_repo_bound_path(repo_root, current_report["batch_path"]),
        "baseline_snapshot": baseline_snapshot["summary_path"],
        "evaluation_summary": evaluation_summary["summary_path"],
        "thresholds": {
            "min_rolling_accuracy": float(thresholds["min_rolling_accuracy"]),
            "max_confidence_drop": float(thresholds["max_confidence_drop"]),
            "max_histogram_js_divergence": float(thresholds["max_histogram_js_divergence"]),
        },
        "baseline_reference": {
            "batch_name": baseline_report["batch_name"],
            "accuracy": float(baseline_report["accuracy"]),
            "macro_f1": float(baseline_report["macro_f1"]),
            "mean_confidence": float(baseline_report["mean_confidence"]),
            "predicted_distribution": list(baseline_report["predicted_distribution"]),
            "histogram_reference_batch_name": histogram_reference["batch_name"],
            "histogram_reference_examples": int(histogram_reference["num_examples"]),
        },
        "production_reference_data": {
            "batch_name": histogram_reference["batch_name"],
            "batch_path": serialize_repo_bound_path(repo_root, histogram_reference["batch_path"]),
            "num_examples": int(histogram_reference["num_examples"]),
        },
        "current_batch": {
            "batch_name": current_report["batch_name"],
            "num_examples": int(current_report["num_examples"]),
            "accuracy": float(current_report["accuracy"]),
            "macro_f1": float(current_report["macro_f1"]),
            "mean_confidence": float(current_report["mean_confidence"]),
            "predicted_distribution": list(current_report["predicted_distribution"]),
            "labels_key": current_report["labels_key"],
        },
        "drift_metrics": {
            "accuracy_drop": accuracy_drop,
            "confidence_drop": confidence_drop,
            "pixel_intensity_histogram_js_divergence": histogram_js_divergence,
            "predicted_class_distribution_js_divergence": predicted_distribution_js_divergence,
        },
        "alerts": alerts,
        "should_alert": bool(alerts),
        "should_retrain": bool(alerts) and int(current_report["num_examples"]) >= min_recent_samples,
        "min_recent_samples": min_recent_samples,
    }

    summary_path = session_dir / "monitoring_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(repo_root))
    return summary


def main(argv: list[str] | None = None) -> int:
    # Expose monitoring analysis through a small command-line interface.
    parser = argparse.ArgumentParser(
        description="Analyze one labeled batch for drift and retraining alerts.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Named batch under data/processed, such as clean_batch_a or drifted_inference_batch.",
    )
    parser.add_argument("--batch-path", default=None, help="Optional explicit path to an .npz batch file.")
    parser.add_argument(
        "--baseline-snapshot-path",
        default=None,
        help="Optional baseline_production_snapshot.json override.",
    )
    parser.add_argument("--output-dir", default="artifacts/monitoring", help="Folder for monitoring reports.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for analyzed examples.")
    parser.add_argument(
        "--use-clean-labels",
        action="store_true",
        help="Use clean_labels when the batch file contains them.",
    )
    args = parser.parse_args(argv)

    summary = analyze_batch_monitoring(
        params_path=args.params,
        batch_name=args.batch_name,
        batch_path=args.batch_path,
        baseline_snapshot_path=args.baseline_snapshot_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        use_clean_labels=args.use_clean_labels,
    )
    print(json.dumps(summary, indent=2))
    return 0


def compute_pixel_histogram_js_divergence(
    repo_root: Path,
    baseline_batch_path: str | Path,
    current_batch_path: str | Path,
    bins: int = 32,
) -> float:
    # Compare baseline and current image histograms with one distance score.
    baseline_images = _load_images(repo_root, baseline_batch_path)
    current_images = _load_images(repo_root, current_batch_path)
    baseline_distribution = _compute_histogram_distribution(baseline_images, bins=bins)
    current_distribution = _compute_histogram_distribution(current_images, bins=bins)
    return compute_js_divergence(baseline_distribution, current_distribution)


def compute_js_divergence(reference_values: list[float] | np.ndarray, current_values: list[float] | np.ndarray) -> float:
    # Compute Jensen-Shannon divergence between two distributions.
    reference = np.asarray(reference_values, dtype=np.float64)
    current = np.asarray(current_values, dtype=np.float64)

    reference = reference / np.clip(reference.sum(), 1e-12, None)
    current = current / np.clip(current.sum(), 1e-12, None)
    reference = np.clip(reference, 1e-12, None)
    current = np.clip(current, 1e-12, None)
    midpoint = 0.5 * (reference + current)

    return float(0.5 * (np.sum(reference * np.log(reference / midpoint)) + np.sum(current * np.log(current / midpoint))))


def _resolve_baseline_snapshot(
    repo_root: Path,
    params_path: Path,
    baseline_snapshot_path: str | Path | None,
) -> dict[str, Any]:
    # Reuse the latest frozen baseline snapshot or create a new one.
    if baseline_snapshot_path is not None:
        snapshot_path = _resolve_path(repo_root, baseline_snapshot_path)
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload = _normalize_report_paths(repo_root, payload)
        payload["summary_path"] = str(snapshot_path.relative_to(repo_root))
        return payload

    evaluation_root = repo_root / "artifacts" / "evaluation"
    candidates = sorted(evaluation_root.glob("*/baseline_production_snapshot.json"))
    if candidates:
        snapshot_path = candidates[-1]
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload = _normalize_report_paths(repo_root, payload)
        payload["summary_path"] = str(snapshot_path.relative_to(repo_root))
        return payload

    return evaluate_registered_model(
        params_path=params_path,
        output_dir="artifacts/evaluation",
        freeze_baseline=True,
    )


def _select_report(summary_payload: dict[str, Any], batch_name: str) -> dict[str, Any]:
    for report in summary_payload.get("reports", []):
        if report["batch_name"] == batch_name:
            return report
    raise ValueError(f"Batch '{batch_name}' not found in monitoring reference summary")


def _select_histogram_reference(summary_payload: dict[str, Any], fallback_report: dict[str, Any]) -> dict[str, Any]:
    # Prefer the training reference data for histogram drift comparisons.
    reference_data = summary_payload.get("production_reference_data")
    if isinstance(reference_data, dict) and reference_data.get("batch_path"):
        return {
            "batch_name": str(reference_data.get("batch_name") or Path(str(reference_data["batch_path"])).stem),
            "batch_path": reference_data["batch_path"],
            "num_examples": int(reference_data.get("num_examples", 0)),
        }

    return {
        "batch_name": fallback_report["batch_name"],
        "batch_path": fallback_report["batch_path"],
        "num_examples": int(fallback_report.get("num_examples", 0)),
    }


def _compute_histogram_distribution(images: np.ndarray, bins: int) -> np.ndarray:
    # Turn image pixels into a normalized histogram distribution.
    histogram, _ = np.histogram(images.reshape(-1), bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    histogram = histogram / np.clip(histogram.sum(), 1e-12, None)
    return np.clip(histogram, 1e-12, None)


def _load_images(repo_root: Path, batch_path: str | Path) -> np.ndarray:
    resolved_path = resolve_repo_bound_path(repo_root, batch_path)
    with np.load(resolved_path) as batch:
        return batch["images"].astype(np.float32)


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    return resolve_repo_bound_path(repo_root, raw_path)


def _normalize_report_paths(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    reference_data = payload.get("production_reference_data")
    if isinstance(reference_data, dict) and reference_data.get("batch_path"):
        reference_data["batch_path"] = serialize_repo_bound_path(repo_root, reference_data["batch_path"])

    for report in payload.get("reports", []):
        if report.get("batch_path"):
            report["batch_path"] = serialize_repo_bound_path(repo_root, report["batch_path"])
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # Persist JSON artifacts with stable formatting.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())