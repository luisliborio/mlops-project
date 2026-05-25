from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from mlops_mnist.utils.mlflow_compat import prepare_local_tracking_uri, resolve_runtime_write_path


def load_runtime_config(repo_root: str | Path) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    config: dict[str, str] = {}

    example_path = repo_root / ".env.example"
    env_path = repo_root / ".env"

    if example_path.exists():
        config.update({key: value for key, value in dotenv_values(example_path).items() if value})
    if env_path.exists():
        config.update({key: value for key, value in dotenv_values(env_path).items() if value})

    return config


def resolve_tracking_uri(repo_root: str | Path, raw_tracking_uri: str) -> str:
    repo_root = Path(repo_root).resolve()
    if raw_tracking_uri.startswith("sqlite:////"):
        return raw_tracking_uri

    if raw_tracking_uri.startswith("sqlite:///"):
        database_path = (repo_root / raw_tracking_uri.removeprefix("sqlite:///" )).resolve()
        return f"sqlite:///{database_path.as_posix()}"

    return raw_tracking_uri


def prepare_runtime_tracking_uri(
    repo_root: str | Path,
    runtime_config: Mapping[str, str],
    default_tracking_uri: str = "sqlite:///mlflow.db",
) -> str:
    tracking_uri = resolve_tracking_uri(
        repo_root,
        runtime_config.get("MLFLOW_TRACKING_URI", default_tracking_uri),
    )
    return prepare_local_tracking_uri(Path(repo_root).resolve(), tracking_uri)


def resolve_artifact_root(
    repo_root: str | Path,
    runtime_config: Mapping[str, str],
    default_artifact_root: str = "./artifacts",
) -> Path:
    repo_root = Path(repo_root).resolve()
    artifact_root = resolve_runtime_write_path(
        repo_root,
        runtime_config.get("MLFLOW_ARTIFACT_ROOT", default_artifact_root),
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def resolve_runtime_port(runtime_config: Mapping[str, str], key: str, default: int) -> int:
    raw_value = runtime_config.get(key)
    if raw_value is None:
        return int(default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(default)