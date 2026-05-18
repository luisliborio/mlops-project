from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy import ndimage

MnistLoader = Callable[
    [], tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
]

# Keep drift settings centralized so params.yaml only selects the severity level.
_DRIFT_SEVERITY_PARAMETERS: dict[str, dict[str, float]] = {
    "low": {
        "noise_std": 0.08,
        "blur_sigma": 0.35,
        "rotation_deg": 8.0,
        "contrast_factor": 0.85,
    },
    "medium": {
        "noise_std": 0.14,
        "blur_sigma": 0.60,
        "rotation_deg": 15.0,
        "contrast_factor": 0.70,
    },
    "high": {
        "noise_std": 0.20,
        "blur_sigma": 0.90,
        "rotation_deg": 25.0,
        "contrast_factor": 0.55,
    },
}


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    # Read the single project config that drives batch sizes and corruption settings.
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    # Convert raw 0-255 grayscale digits into normalized CNN-ready tensors.
    normalized = images.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]


def apply_drift_transforms(
    images: np.ndarray,
    transforms: list[str],
    severity: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if severity not in _DRIFT_SEVERITY_PARAMETERS:
        raise ValueError(f"Unsupported drift severity: {severity}")

    # Apply the selected corruptions in order so the output is reproducible.
    parameters = _DRIFT_SEVERITY_PARAMETERS[severity]
    drifted_images = images.astype(np.float32, copy=True)

    for transform in transforms:
        if transform == "gaussian_noise":
            # Add pixel-level random perturbations.
            noise = rng.normal(
                loc=0.0,
                scale=parameters["noise_std"],
                size=drifted_images.shape,
            ).astype(np.float32)
            drifted_images = np.clip(drifted_images + noise, 0.0, 1.0)
            continue

        if transform == "gaussian_blur":
            # Blur the spatial dimensions while leaving batch and channel axes untouched.
            drifted_images = ndimage.gaussian_filter(
                drifted_images,
                sigma=(0.0, parameters["blur_sigma"], parameters["blur_sigma"], 0.0),
                mode="nearest",
            ).astype(np.float32)
            continue

        if transform == "rotation":
            # Rotate each image independently without changing the 28x28 canvas size.
            rotated = np.empty_like(drifted_images)
            for index, image in enumerate(drifted_images):
                rotated[index, ..., 0] = ndimage.rotate(
                    image[..., 0],
                    angle=parameters["rotation_deg"],
                    reshape=False,
                    order=1,
                    mode="nearest",
                )
            drifted_images = np.clip(rotated, 0.0, 1.0)
            continue

        if transform == "contrast_shift":
            # Pull each image closer to its mean intensity to reduce contrast.
            image_means = drifted_images.mean(axis=(1, 2, 3), keepdims=True)
            drifted_images = np.clip(
                (drifted_images - image_means) * parameters["contrast_factor"]
                + image_means,
                0.0,
                1.0,
            ).astype(np.float32)
            continue

        raise ValueError(f"Unsupported drift transform: {transform}")

    return drifted_images.astype(np.float32)


def generate_noisy_labels(
    labels: np.ndarray,
    flip_fraction: float,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= flip_fraction <= 1.0:
        raise ValueError("flip_fraction must be between 0.0 and 1.0")

    # Flip a deterministic subset of labels to simulate bad retraining data.
    noisy_labels = labels.astype(np.int64, copy=True)
    flip_count = int(round(len(labels) * flip_fraction))
    flipped_mask = np.zeros(len(labels), dtype=bool)

    if flip_count == 0:
        return noisy_labels, flipped_mask

    indices = rng.choice(len(labels), size=flip_count, replace=False)
    flipped_mask[indices] = True

    for index in indices:
        current_label = int(noisy_labels[index])
        candidates = np.delete(np.arange(num_classes, dtype=np.int64), current_label)
        noisy_labels[index] = int(rng.choice(candidates))

    return noisy_labels, flipped_mask


def create_data_artifacts(
    params_path: str | Path = "params.yaml",
    output_root: str | Path | None = None,
    mnist_loader: MnistLoader | None = None,
) -> dict[str, Any]:
    # Run both stage functions in sequence for the original one-command workflow.
    prepare_baseline_artifacts(
        params_path=params_path,
        output_root=output_root,
        mnist_loader=mnist_loader,
    )
    return generate_scenario_artifacts(
        params_path=params_path,
        output_root=output_root,
    )


def prepare_baseline_artifacts(
    params_path: str | Path = "params.yaml",
    output_root: str | Path | None = None,
    mnist_loader: MnistLoader | None = None,
) -> dict[str, Any]:
    # First DVC stage: create the clean baseline data and the clean source batches.
    params_path = Path(params_path).resolve()
    params = load_project_params(params_path)
    data_root, raw_dir, interim_dir, processed_dir = _resolve_pipeline_paths(
        params_path,
        output_root,
    )

    # Split one base seed into dedicated RNGs so each step stays reproducible.
    seed = int(params["project"]["random_seed"])
    train_rng, test_rng, _, _ = _spawn_rngs(seed)

    if mnist_loader is None:
        mnist_loader = _default_mnist_loader

    (train_images, train_labels), (test_images, test_labels) = mnist_loader()
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)

    # Save an untouched raw snapshot before any shuffling or slicing happens.
    raw_snapshot_path = raw_dir / "mnist.npz"
    _save_npz(
        raw_snapshot_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    train_images, train_labels = _shuffle_pairs(train_images, train_labels, train_rng)
    test_images, test_labels = _shuffle_pairs(test_images, test_labels, test_rng)

    baseline_config = params["data"]["baseline"]
    future_batches_config = params["data"]["future_batches"]
    num_classes = int(params["model"]["num_classes"])

    # Consume the shuffled arrays sequentially so batches never overlap by accident.
    train_cursor = 0
    test_cursor = 0

    baseline_train, train_cursor = _take_batch(
        train_images,
        train_labels,
        train_cursor,
        int(baseline_config["train_size"]),
        "baseline_train",
    )
    baseline_validation, train_cursor = _take_batch(
        train_images,
        train_labels,
        train_cursor,
        int(baseline_config["validation_size"]),
        "baseline_validation",
    )
    clean_batch_a, train_cursor = _take_batch(
        train_images,
        train_labels,
        train_cursor,
        int(future_batches_config["clean_batch_a_size"]),
        "clean_batch_a",
    )
    clean_batch_b, train_cursor = _take_batch(
        train_images,
        train_labels,
        train_cursor,
        int(future_batches_config["clean_batch_b_size"]),
        "clean_batch_b",
    )
    noisy_retraining_source_batch, train_cursor = _take_batch(
        train_images,
        train_labels,
        train_cursor,
        int(future_batches_config["noisy_retraining_batch_size"]),
        "noisy_retraining_source_batch",
    )

    baseline_test, test_cursor = _take_batch(
        test_images,
        test_labels,
        test_cursor,
        int(baseline_config["test_size"]),
        "baseline_test",
    )
    drift_source_batch, test_cursor = _take_batch(
        test_images,
        test_labels,
        test_cursor,
        int(future_batches_config["drifted_inference_batch_size"]),
        "drift_source_batch",
    )

    manifest: dict[str, Any] = {
        "project_name": params["project"]["name"],
        "seed": seed,
        "raw_snapshot": _build_artifact_metadata(raw_snapshot_path, data_root),
        "batches": {},
        "interim_sources": {},
    }

    # Write the clean baseline and clean future batches first.
    manifest["batches"]["baseline_train"] = _write_processed_batch(
        processed_dir / "baseline_train.npz",
        preprocess_images(baseline_train[0]),
        baseline_train[1],
        data_root,
        num_classes,
    )
    manifest["batches"]["baseline_validation"] = _write_processed_batch(
        processed_dir / "baseline_validation.npz",
        preprocess_images(baseline_validation[0]),
        baseline_validation[1],
        data_root,
        num_classes,
    )
    manifest["batches"]["baseline_test"] = _write_processed_batch(
        processed_dir / "baseline_test.npz",
        preprocess_images(baseline_test[0]),
        baseline_test[1],
        data_root,
        num_classes,
    )
    manifest["batches"]["clean_batch_a"] = _write_processed_batch(
        processed_dir / "clean_batch_a.npz",
        preprocess_images(clean_batch_a[0]),
        clean_batch_a[1],
        data_root,
        num_classes,
    )
    manifest["batches"]["clean_batch_b"] = _write_processed_batch(
        processed_dir / "clean_batch_b.npz",
        preprocess_images(clean_batch_b[0]),
        clean_batch_b[1],
        data_root,
        num_classes,
    )

    # Save clean inputs for the second DVC stage before applying any corruption.
    manifest["interim_sources"]["drift_source_batch"] = _write_processed_batch(
        interim_dir / "drift_source_batch.npz",
        preprocess_images(drift_source_batch[0]),
        drift_source_batch[1],
        data_root,
        num_classes,
    )
    manifest["interim_sources"]["noisy_retraining_source_batch"] = _write_processed_batch(
        interim_dir / "noisy_retraining_source_batch.npz",
        preprocess_images(noisy_retraining_source_batch[0]),
        noisy_retraining_source_batch[1],
        data_root,
        num_classes,
    )

    # This base manifest becomes the handoff between the two DVC stages.
    manifest_path = interim_dir / "base_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = _relative_path(manifest_path, data_root)

    return manifest


def generate_scenario_artifacts(
    params_path: str | Path = "params.yaml",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    # Second DVC stage: turn clean source batches into drifted and noisy scenarios.
    params_path = Path(params_path).resolve()
    params = load_project_params(params_path)
    data_root, _, interim_dir, processed_dir = _resolve_pipeline_paths(
        params_path,
        output_root,
    )

    base_manifest_path = interim_dir / "base_manifest.json"
    if not base_manifest_path.exists():
        raise FileNotFoundError(
            "Missing data/interim/base_manifest.json. "
            "Run the prepare-baseline stage first."
        )

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    seed = int(params["project"]["random_seed"])
    _, _, drift_rng, label_rng = _spawn_rngs(seed)

    drift_config = params["data"]["drift"]
    noisy_labels_config = params["data"]["noisy_labels"]
    num_classes = int(params["model"]["num_classes"])

    drift_source_images, drift_source_labels = _load_saved_batch(
        interim_dir / "drift_source_batch.npz"
    )
    noisy_source_images, noisy_source_labels = _load_saved_batch(
        interim_dir / "noisy_retraining_source_batch.npz"
    )

    manifest: dict[str, Any] = {
        "project_name": base_manifest["project_name"],
        "seed": seed,
        "raw_snapshot": base_manifest["raw_snapshot"],
        "batches": dict(base_manifest["batches"]),
        "interim_sources": base_manifest.get("interim_sources", {}),
    }

    # Drift is applied only after the clean source batch has been versioned.
    drifted_images = apply_drift_transforms(
        drift_source_images,
        list(drift_config["transforms"]),
        str(drift_config["severity"]),
        drift_rng,
    )
    manifest["batches"]["drifted_inference_batch"] = _write_processed_batch(
        processed_dir / "drifted_inference_batch.npz",
        drifted_images,
        drift_source_labels,
        data_root,
        num_classes,
        transforms=list(drift_config["transforms"]),
        severity=str(drift_config["severity"]),
    )

    # Keep both clean and noisy labels for later evaluation and debugging.
    noisy_labels, flipped_mask = generate_noisy_labels(
        noisy_source_labels,
        float(noisy_labels_config["flip_fraction"]),
        num_classes,
        label_rng,
    )
    manifest["batches"]["noisy_retraining_batch"] = _write_processed_batch(
        processed_dir / "noisy_retraining_batch.npz",
        noisy_source_images,
        noisy_labels,
        data_root,
        num_classes,
        clean_labels=noisy_source_labels,
        flipped_mask=flipped_mask,
        flip_fraction=float(noisy_labels_config["flip_fraction"]),
    )

    # The final manifest summarizes all clean and simulated outputs.
    manifest_path = interim_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = _relative_path(manifest_path, data_root)

    return manifest


def main(argv: list[str] | None = None) -> int:
    # Keep one CLI entrypoint, but let DVC call each stage separately.
    parser = argparse.ArgumentParser(
        description="Prepare deterministic MNIST data artifacts for the Day 2 and Day 3 pipeline.",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to the project params.yaml file.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional alternate data directory; defaults to ./data.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prepare-baseline", "generate-scenarios"],
        default="all",
        help="Select one DVC-friendly stage or run the full pipeline.",
    )
    args = parser.parse_args(argv)

    if args.stage == "prepare-baseline":
        manifest = prepare_baseline_artifacts(
            params_path=args.params,
            output_root=args.output_root,
        )
    elif args.stage == "generate-scenarios":
        manifest = generate_scenario_artifacts(
            params_path=args.params,
            output_root=args.output_root,
        )
    else:
        manifest = create_data_artifacts(
            params_path=args.params,
            output_root=args.output_root,
        )

    summary = {
        "manifest_path": manifest["manifest_path"],
        "batches": {
            name: batch["num_examples"] for name, batch in manifest["batches"].items()
        },
    }
    if "interim_sources" in manifest:
        summary["interim_sources"] = {
            name: batch["num_examples"]
            for name, batch in manifest["interim_sources"].items()
        }
    print(json.dumps(summary, indent=2))
    return 0


def _default_mnist_loader() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    # Import TensorFlow lazily so tests can inject a fake loader without paying that cost.
    from tensorflow.keras.datasets import mnist

    return mnist.load_data()


def _resolve_pipeline_paths(
    params_path: Path,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    data_root = Path(output_root).resolve() if output_root else params_path.parent / "data"
    raw_dir = data_root / "raw"
    interim_dir = data_root / "interim"
    processed_dir = data_root / "processed"

    for directory in (raw_dir, interim_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return data_root, raw_dir, interim_dir, processed_dir


def _spawn_rngs(
    seed: int,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator]:
    seed_sequence = np.random.SeedSequence(seed)
    generators = [np.random.default_rng(child_seed) for child_seed in seed_sequence.spawn(4)]
    return tuple(generators)


def _shuffle_pairs(
    images: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Shuffle images and labels together to preserve their alignment.
    indices = rng.permutation(len(images))
    return images[indices], labels[indices]


def _take_batch(
    images: np.ndarray,
    labels: np.ndarray,
    start_index: int,
    batch_size: int,
    batch_name: str,
) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    # Slice the next contiguous batch and advance the cursor.
    end_index = start_index + batch_size
    if end_index > len(images):
        raise ValueError(
            f"Not enough samples to create {batch_name}: "
            f"requested {batch_size}, available {len(images) - start_index}"
        )
    return (images[start_index:end_index], labels[start_index:end_index]), end_index


def _write_processed_batch(
    output_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    data_root: Path,
    num_classes: int,
    **extra_arrays_or_metadata: Any,
) -> dict[str, Any]:
    # Store arrays in the NPZ file and non-array fields in the manifest metadata.
    arrays_to_write: dict[str, Any] = {
        "images": images.astype(np.float32),
        "labels": labels.astype(np.int64),
    }
    metadata: dict[str, Any] = {}

    for key, value in extra_arrays_or_metadata.items():
        if isinstance(value, np.ndarray):
            arrays_to_write[key] = value
            continue
        metadata[key] = value

    _save_npz(output_path, **arrays_to_write)

    artifact_metadata = _build_artifact_metadata(output_path, data_root)
    artifact_metadata.update(
        {
            "num_examples": int(labels.shape[0]),
            "image_shape": list(images.shape[1:]),
            "label_distribution": np.bincount(labels, minlength=num_classes).tolist(),
        }
    )
    artifact_metadata.update(metadata)
    return artifact_metadata


def _load_saved_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required batch file: {path}")

    with np.load(path) as batch:
        images = batch["images"].astype(np.float32)
        labels = batch["labels"].astype(np.int64)

    return images, labels


def _save_npz(path: Path, **arrays: Any) -> None:
    # Keep each generated artifact as a compressed numpy archive for simple reuse.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _build_artifact_metadata(path: Path, data_root: Path) -> dict[str, Any]:
    # Hashes and sizes make later DVC or debugging checks easier.
    return {
        "path": _relative_path(path, data_root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())