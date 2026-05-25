from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from tensorflow import keras

from mlops_mnist.utils.mlflow_compat import prepare_local_tracking_uri, resolve_runtime_write_path
from mlops_mnist.utils.runtime import load_runtime_config, resolve_artifact_root, resolve_tracking_uri


EARLY_STOPPING_PATIENCE = 5


def load_project_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_candidates(
    params_path: str | Path = "params.yaml",
    data_root: str | Path | None = None,
    output_dir: str | Path = "artifacts/training",
    models_dir: str | Path = "models/training",
    candidate_name: str | None = None,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> dict[str, Any]:
    # Prefer project-local MLflow settings so runs stay isolated to this repo.
    params_path = Path(params_path).resolve()
    repo_root = params_path.parent
    params = load_project_params(params_path)
    runtime_config = _load_runtime_config(repo_root)

    resolved_data_root = _resolve_path(
        repo_root,
        data_root or runtime_config.get("DATA_ROOT", "data"),
    )
    resolved_output_dir = resolve_runtime_write_path(repo_root, output_dir)
    resolved_models_dir = resolve_runtime_write_path(repo_root, models_dir)
    manifest_path = resolved_data_root / "interim" / "data_manifest.json"

    datasets = _load_datasets(
        resolved_data_root,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )

    candidates = list(params["model"]["candidates"])
    if candidate_name is not None:
        candidates = [candidate for candidate in candidates if candidate["name"] == candidate_name]
        if not candidates:
            available = ", ".join(candidate["name"] for candidate in params["model"]["candidates"])
            raise ValueError(f"Unknown candidate '{candidate_name}'. Available: {available}")

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
    tracking_uri = prepare_local_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_name)

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = resolved_output_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    image_shape = tuple(params["model"]["image_shape"])
    num_classes = int(params["model"]["num_classes"])
    base_seed = int(params["project"]["random_seed"])
    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        # Reset Keras state before each candidate run.
        keras.backend.clear_session()
        keras.utils.set_random_seed(base_seed + index)

        candidate_dir = session_dir / candidate["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolved_models_dir / session_name / f"{candidate['name']}.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model = build_cnn_model(
            candidate_config=candidate,
            image_shape=image_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
        )

        run_name = f"{candidate['name']}-{session_name}"
        with mlflow.start_run(run_name=run_name) as run:
            _log_run_params(
                candidate=candidate,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                params_path=params_path,
                manifest_path=manifest_path,
                datasets=datasets,
            )

            # Stop once validation loss stops improving and keep the best weights.
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )

            history = model.fit(
                datasets["x_train"],
                datasets["y_train"],
                validation_data=(datasets["x_validation"], datasets["y_validation"]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=2,
            )

            metrics = _evaluate_model(
                model=model,
                history=history.history,
                early_stopping=early_stopping,
                x_validation=datasets["x_validation"],
                y_validation=datasets["y_validation"],
                x_test=datasets["x_test"],
                y_test=datasets["y_test"],
            )

            model.save(model_path)
            history_path = candidate_dir / "history.json"
            metrics_path = candidate_dir / "metrics.json"
            _write_json(history_path, history.history)
            _write_json(metrics_path, metrics)

            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.tensorflow.log_model(model, artifact_path="model")
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(history_path), artifact_path="reports")
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
            if manifest_path.exists():
                mlflow.log_artifact(str(manifest_path), artifact_path="data")

            results.append(
                {
                    "candidate_name": candidate["name"],
                    "run_id": run.info.run_id,
                    "model_uri": model_uri,
                    "model_path": str(model_path.relative_to(repo_root)),
                    **metrics,
                }
            )

    comparison_csv_path = session_dir / "comparison.csv"
    summary_json_path = session_dir / "summary.json"
    summary_md_path = session_dir / "summary.md"
    _write_comparison_csv(comparison_csv_path, results)
    _write_json(summary_json_path, {"results": results})
    _write_summary_markdown(summary_md_path, results, experiment_name)

    best_result = max(results, key=lambda item: item["best_validation_accuracy"])
    return {
        "session_dir": str(session_dir.relative_to(repo_root)),
        "comparison_csv": str(comparison_csv_path.relative_to(repo_root)),
        "summary_json": str(summary_json_path.relative_to(repo_root)),
        "summary_md": str(summary_md_path.relative_to(repo_root)),
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "artifact_root": str(artifact_root.relative_to(repo_root)),
        "best_candidate": best_result["candidate_name"],
        "best_validation_accuracy": best_result["best_validation_accuracy"],
        "runs": results,
    }


def build_cnn_model(
    candidate_config: dict[str, Any],
    image_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    # Build one small CNN from the candidate definition in params.yaml.
    model = keras.Sequential(name=candidate_config["name"])
    model.add(keras.layers.Input(shape=image_shape))

    for filters in candidate_config["conv_filters"]:
        model.add(keras.layers.Conv2D(int(filters), kernel_size=3, padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(int(candidate_config["dense_units"]), activation="relu"))
    model.add(keras.layers.Dropout(float(candidate_config["dropout"])))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN candidates and log the results to MLflow.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--data-root", default=None, help="Optional path to the local data directory.")
    parser.add_argument("--output-dir", default="artifacts/training", help="Folder for local reports.")
    parser.add_argument("--models-dir", default="models/training", help="Folder for saved Keras models.")
    parser.add_argument("--candidate", default=None, help="Train only one candidate from params.yaml.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap for train examples.")
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Optional cap for validation examples.",
    )
    parser.add_argument("--test-limit", type=int, default=None, help="Optional cap for test examples.")
    args = parser.parse_args(argv)

    summary = train_candidates(
        params_path=args.params,
        data_root=args.data_root,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        candidate_name=args.candidate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    return load_runtime_config(repo_root)


def _resolve_tracking_uri(repo_root: Path, raw_tracking_uri: str) -> str:
    return resolve_tracking_uri(repo_root, raw_tracking_uri)


def _resolve_artifact_root(repo_root: Path, raw_artifact_root: str) -> Path:
    return resolve_artifact_root(repo_root, {"MLFLOW_ARTIFACT_ROOT": raw_artifact_root})


def _ensure_experiment(experiment_name: str, artifact_root: Path) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())


def _load_datasets(
    data_root: Path,
    train_limit: int | None,
    validation_limit: int | None,
    test_limit: int | None,
) -> dict[str, np.ndarray]:
    processed_dir = data_root / "processed"
    train_path = processed_dir / "baseline_train.npz"
    validation_path = processed_dir / "baseline_validation.npz"
    test_path = processed_dir / "baseline_test.npz"

    missing_paths = [path for path in (train_path, validation_path, test_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing processed dataset files: {missing_text}. "
            "Run 'dvc repro' or 'python scripts/prepare_data.py' first."
        )

    x_train, y_train = _load_split(train_path, train_limit)
    x_validation, y_validation = _load_split(validation_path, validation_limit)
    x_test, y_test = _load_split(test_path, test_limit)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": y_test,
    }


def _load_split(path: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as batch:
        images = batch["images"].astype(np.float32)
        labels = batch["labels"].astype(np.int64)

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    return images, labels


def _log_run_params(
    candidate: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    params_path: Path,
    manifest_path: Path,
    datasets: dict[str, np.ndarray],
) -> None:
    # Keep the MLflow run metadata readable and close to the CLI inputs.
    mlflow.log_params(
        {
            "candidate_name": candidate["name"],
            "conv_filters": ",".join(str(filters) for filters in candidate["conv_filters"]),
            "dense_units": int(candidate["dense_units"]),
            "dropout": float(candidate["dropout"]),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "params_path": str(params_path.name),
            "data_manifest": str(manifest_path.relative_to(params_path.parent)) if manifest_path.exists() else "missing",
            "train_examples": len(datasets["x_train"]),
            "validation_examples": len(datasets["x_validation"]),
            "test_examples": len(datasets["x_test"]),
        }
    )


def _evaluate_model(
    model: keras.Model,
    history: dict[str, list[float]],
    early_stopping: keras.callbacks.EarlyStopping,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    validation_loss, validation_accuracy = model.evaluate(x_validation, y_validation, verbose=0)
    validation_predictions = np.argmax(model.predict(x_validation, verbose=0), axis=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    best_validation_accuracy = float(history["val_accuracy"][best_epoch - 1])

    return {
        "final_train_loss": float(history["loss"][-1]),
        "final_train_accuracy": float(history["accuracy"][-1]),
        "final_validation_loss": float(validation_loss),
        "final_validation_accuracy": float(validation_accuracy),
        "best_validation_loss": float(min(history["val_loss"])),
        "best_validation_accuracy": best_validation_accuracy,
        "best_epoch": float(best_epoch),
        "trained_epochs": float(len(history["loss"])),
        "stopped_early": float(1.0 if early_stopping.stopped_epoch > 0 else 0.0),
        "validation_macro_f1": float(f1_score(y_validation, validation_predictions, average="macro")),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
    }


def _write_comparison_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_name",
        "run_id",
        "model_uri",
        "best_validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "model_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fieldnames})


def _write_summary_markdown(
    path: Path,
    results: list[dict[str, Any]],
    experiment_name: str,
) -> None:
    best_result = max(results, key=lambda item: item["best_validation_accuracy"])
    lines = [
        "# Training Summary",
        "",
        f"Experiment: `{experiment_name}`",
        "",
        f"Best candidate: `{best_result['candidate_name']}`",
        f"Best validation accuracy: `{best_result['best_validation_accuracy']:.4f}`",
        "",
        "| candidate | best val acc | val macro F1 | test acc | test macro F1 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result['candidate_name']} | "
            f"{result['best_validation_accuracy']:.4f} | "
            f"{result['validation_macro_f1']:.4f} | "
            f"{result['test_accuracy']:.4f} | "
            f"{result['test_macro_f1']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())