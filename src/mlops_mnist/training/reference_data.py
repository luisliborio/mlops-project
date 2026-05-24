from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mlops_mnist.utils.mlflow_compat import (
    resolve_repo_bound_path,
    resolve_runtime_write_path,
    serialize_repo_bound_path,
)


DEFAULT_TRAINING_REFERENCE_BATCH = "baseline_train"
TRAINING_REFERENCE_NAME_TAG = "training_reference_data_name"
TRAINING_REFERENCE_PATH_TAG = "training_reference_data_path"
TRAINING_REFERENCE_SIZE_TAG = "training_reference_data_examples"


def resolve_training_reference_metadata(
    repo_root: str | Path,
    version_tags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    resolved_path = _resolve_training_reference_path(repo_root, version_tags)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Training reference dataset not found: {resolved_path}")

    with np.load(resolved_path) as batch:
        num_examples = int(len(batch["images"]))

    tags = version_tags or {}
    batch_name = str(tags.get(TRAINING_REFERENCE_NAME_TAG) or resolved_path.stem)

    return {
        "batch_name": batch_name,
        "batch_path": serialize_repo_bound_path(repo_root, resolved_path),
        "num_examples": num_examples,
    }


def load_training_reference_data(
    repo_root: str | Path,
    version_tags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = resolve_training_reference_metadata(repo_root, version_tags)
    repo_root = Path(repo_root).resolve()
    resolved_path = resolve_repo_bound_path(repo_root, metadata["batch_path"])

    with np.load(resolved_path) as batch:
        images = batch["images"].astype(np.float32)
        labels = batch["labels"].astype(np.int64)

    if len(images) != len(labels):
        raise ValueError("Training reference dataset images and labels must have matching lengths")

    return {
        **metadata,
        "images": images,
        "labels": labels,
    }


def materialize_training_reference_data(
    repo_root: str | Path,
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    batch_name: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    if len(images) != len(labels):
        raise ValueError("Training reference images and labels must have matching lengths")
    resolved_output_path = resolve_runtime_write_path(repo_root, output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        resolved_output_path,
        images=np.asarray(images, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )

    return {
        "batch_name": batch_name or resolved_output_path.stem,
        "batch_path": serialize_repo_bound_path(repo_root, resolved_output_path),
        "num_examples": int(len(images)),
    }


def _resolve_training_reference_path(
    repo_root: Path,
    version_tags: Mapping[str, Any] | None,
) -> Path:
    tags = version_tags or {}
    tagged_path = tags.get(TRAINING_REFERENCE_PATH_TAG)
    if tagged_path:
        return resolve_repo_bound_path(repo_root, str(tagged_path))

    return (repo_root / "data" / "processed" / f"{DEFAULT_TRAINING_REFERENCE_BATCH}.npz").resolve()