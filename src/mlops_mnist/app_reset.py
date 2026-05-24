from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import yaml
from dotenv import dotenv_values
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient, _tracking_service, fluent
from mlflow.tracking._model_registry import utils as _model_registry_utils
from sqlalchemy.orm import close_all_sessions
from tensorflow import keras

from mlops_mnist.training.reference_data import (
    TRAINING_REFERENCE_NAME_TAG,
    TRAINING_REFERENCE_PATH_TAG,
    TRAINING_REFERENCE_SIZE_TAG,
    resolve_training_reference_metadata,
)
from mlops_mnist.ui.lifecycle import reset_lifecycle_state
from mlops_mnist.utils.mlflow_compat import prepare_local_tracking_uri, resolve_runtime_write_path


APP_RESET_SEED_PATH = Path("configs/app_reset_seed.json")
RESET_REPORT_RELATIVE_PATH = Path("artifacts/registry/reset_to_initial_state/promotion_summary.json")
HEX_ARTIFACT_DIR_RE = re.compile(r"^[0-9a-f]{32}$")
GENERATED_PATHS = (
    Path(".runtime"),
    Path("mlflow.db"),
    Path("mlruns"),
    Path("artifacts/evaluation"),
    Path("artifacts/evaluation_smoke_fix"),
    Path("artifacts/models"),
    Path("artifacts/monitoring"),
    Path("artifacts/monitoring_checks"),
    Path("artifacts/monitoring_cli_checks"),
    Path("artifacts/registry"),
    Path("artifacts/retraining"),
    Path("artifacts/smoke"),
    Path("artifacts/ui_lifecycle"),
    Path("logs/predictions"),
    Path("models/retraining"),
)


def reset_application_state(
    repo_root: str | Path,
    params_path: str | Path = "params.yaml",
    seed_path: str | Path = APP_RESET_SEED_PATH,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    params_path = _resolve_path(repo_root, params_path)
    seed = _load_reset_seed(repo_root, seed_path)

    _reset_mlflow_runtime_state()
    removed_paths = _clear_generated_state(repo_root)
    mlflow_reset = _bootstrap_initial_mlflow_state(repo_root, params_path, seed)
    lifecycle_state = reset_lifecycle_state(
        repo_root=repo_root,
        batch_name=str(seed["default_batch_name"]),
        batch_size=int(seed["default_batch_size"]),
        baseline_snapshot_path=None,
        baseline_reference=None,
        record_mode=bool(seed.get("default_record_mode", False)),
    )

    return {
        "removed_paths": removed_paths,
        "mlflow_reset": mlflow_reset,
        "lifecycle": {
            "active_batch_name": lifecycle_state["active_batch_name"],
            "batch_size": lifecycle_state["batch_size"],
            "record_mode": lifecycle_state["record_mode"],
            "state_path": str((repo_root / "artifacts" / "ui_lifecycle" / "active_session.json").relative_to(repo_root)),
        },
    }


def ensure_initial_production_alias(
    repo_root: str | Path,
    params_path: str | Path = "params.yaml",
    seed_path: str | Path = APP_RESET_SEED_PATH,
) -> dict[str, Any] | None:
    repo_root = Path(repo_root).resolve()
    params_path = _resolve_path(repo_root, params_path)
    seed = _load_reset_seed(repo_root, seed_path)
    params = _load_project_params(params_path)
    runtime_config = _load_runtime_config(repo_root)

    tracking_uri = _resolve_tracking_uri(
        repo_root,
        runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    tracking_uri = prepare_local_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    _reset_mlflow_runtime_state()
    client = MlflowClient()

    registered_model_name = str(seed.get("registered_model_name") or params["project"]["registered_model_name"])
    alias = str(seed.get("alias", "production"))

    try:
        client.get_model_version_by_alias(registered_model_name, alias)
        return None
    except MlflowException as exc:
        if not _is_missing_production_alias_error(exc):
            raise

    existing_versions = list(client.search_model_versions(f"name = '{registered_model_name}'"))
    if existing_versions:
        return None

    return _bootstrap_initial_mlflow_state(repo_root, params_path, seed)


def _bootstrap_initial_mlflow_state(
    repo_root: Path,
    params_path: Path,
    seed: dict[str, Any],
) -> dict[str, Any]:
    params = _load_project_params(params_path)
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
    registered_model_name = str(seed.get("registered_model_name") or params["project"]["registered_model_name"])
    alias = str(seed.get("alias", "production"))
    seed_model_path = _resolve_path(repo_root, seed["seed_model_path"])
    if not seed_model_path.exists():
        raise FileNotFoundError(f"Initial seed model not found: {seed_model_path}")

    training_reference = resolve_training_reference_metadata(
        repo_root,
        {
            TRAINING_REFERENCE_NAME_TAG: seed.get("training_reference_name"),
            TRAINING_REFERENCE_PATH_TAG: seed.get("training_reference_path"),
        },
    )

    _reset_mlflow_runtime_state()
    tracking_uri = prepare_local_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    model = keras.models.load_model(seed_model_path)

    report_path = resolve_runtime_write_path(repo_root, RESET_REPORT_RELATIVE_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = f"restore-initial-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "reset_seed_model_path": str(seed_model_path.relative_to(repo_root)),
                "registered_model_name": registered_model_name,
                "registered_alias": alias,
                "source_training_run_id": str(seed["source_training_run_id"]),
                "source_summary": str(seed["source_summary"]),
                "candidate_name": str(seed["candidate_name"]),
                "reset_origin": "full_system_reset",
            }
        )
        mlflow.tensorflow.log_model(model, artifact_path="model")

        registered_version = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
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
            value=str(seed["source_training_run_id"]),
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=registered_version.version,
            key="candidate_name",
            value=str(seed["candidate_name"]),
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

        summary = {
            "promotion_run_id": run.info.run_id,
            "registered_model_name": registered_model_name,
            "registered_version": int(registered_version.version),
            "alias": alias,
            "tracking_uri": tracking_uri,
            "selected_result": {
                "candidate_name": str(seed["candidate_name"]),
                "run_id": str(seed["source_training_run_id"]),
                "model_path": str(seed_model_path.relative_to(repo_root)),
            },
            "training_reference": training_reference,
            "source_summary": str(seed["source_summary"]),
            "reset_origin": "full_system_reset",
        }
        _write_json(report_path, summary)

    if int(registered_version.version) != 1:
        raise RuntimeError(
            f"Full reset expected to recreate production version 1, got version {registered_version.version}"
        )

    return {
        **summary,
        "report_path": str(report_path.relative_to(repo_root)),
    }


def _clear_generated_state(repo_root: Path) -> list[str]:
    removed_paths: list[str] = []
    for relative_path in GENERATED_PATHS:
        candidate = (repo_root / relative_path).resolve()
        if _remove_path(candidate):
            removed_paths.append(str(candidate.relative_to(repo_root)))

    artifacts_root = (repo_root / "artifacts").resolve()
    if artifacts_root.exists():
        for child in artifacts_root.iterdir():
            if child.is_dir() and HEX_ARTIFACT_DIR_RE.fullmatch(child.name):
                if _remove_path(child):
                    removed_paths.append(str(child.relative_to(repo_root)))

    return sorted(removed_paths)


def _load_reset_seed(repo_root: Path, seed_path: str | Path) -> dict[str, Any]:
    resolved_seed_path = _resolve_path(repo_root, seed_path)
    if not resolved_seed_path.exists():
        raise FileNotFoundError(f"Application reset seed not found: {resolved_seed_path}")
    return json.loads(resolved_seed_path.read_text(encoding="utf-8"))


def _reset_mlflow_runtime_state() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()

    # Close ORM sessions and dispose pooled DB connections before deleting/recreating sqlite files.
    close_all_sessions()
    _dispose_store_engine(getattr(_tracking_service.utils, "_get_store", None))
    _dispose_store_engine(getattr(_model_registry_utils, "_get_store", None))

    _reset_thread_local_or_collection(getattr(fluent, "_active_run_stack", None), default=[])
    _reset_thread_local_or_collection(getattr(fluent, "_last_active_run_id", None), default=None)
    if hasattr(fluent, "_active_experiment_id"):
        fluent._active_experiment_id = None

    store_registry = getattr(_tracking_service.utils, "_tracking_store_registry", None)
    cache_clear = getattr(getattr(store_registry, "_get_store_with_resolved_uri", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    registry_store_registry_getter = getattr(_model_registry_utils, "_get_store_registry", None)
    if callable(registry_store_registry_getter):
        registry_store_registry = registry_store_registry_getter()
        registry_cache_clear = getattr(
            getattr(registry_store_registry, "_get_store_with_resolved_uri", None),
            "cache_clear",
            None,
        )
        if callable(registry_cache_clear):
            registry_cache_clear()

    artifact_cache = getattr(_tracking_service.utils, "_artifact_repos_cache", None)
    if hasattr(artifact_cache, "clear"):
        artifact_cache.clear()


def _dispose_store_engine(store_getter: Any) -> None:
    if not callable(store_getter):
        return

    try:
        store = store_getter()
    except Exception:
        return

    engine = getattr(store, "engine", None)
    dispose = getattr(engine, "dispose", None)
    if callable(dispose):
        dispose()


def _reset_thread_local_or_collection(state_holder: Any, default: Any) -> None:
    if state_holder is None:
        return

    if hasattr(state_holder, "reset"):
        state_holder.reset()
        return

    if hasattr(state_holder, "clear"):
        state_holder.clear()
        return

    if hasattr(state_holder, "set"):
        state_holder.set(default)


def _load_project_params(params_path: Path) -> dict[str, Any]:
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def _resolve_artifact_root(repo_root: Path, raw_artifact_root: str) -> Path:
    artifact_root = resolve_runtime_write_path(repo_root, raw_artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def _ensure_experiment(experiment_name: str, artifact_root: Path) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())


def _wait_for_model_version(client: MlflowClient, model_name: str, version: str | int) -> None:
    model_version = client.get_model_version(model_name, str(version))
    status = model_version.status.upper()
    if status == "READY":
        return
    if status == "FAILED_REGISTRATION":
        raise RuntimeError(f"Model version {model_name} v{version} failed to register")


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _is_missing_production_alias_error(exc: Exception) -> bool:
    if not isinstance(exc, MlflowException):
        return False

    message = str(exc).lower()
    return "registered model alias" in message and "not found" in message


def _remove_path(path: Path) -> bool:
    if not path.exists():
        return False

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")