from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse


RUNTIME_ROOT_RELATIVE_PATH = Path(".runtime")

TRACKING_PATH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("experiments", "artifact_location"),
    ("logged_model_tags", "tag_value"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("runs", "artifact_uri"),
    ("tags", "value"),
)


def prepare_local_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    """Return a writable local MLflow SQLite URI seeded from container-generated metadata."""
    database_path = _resolve_sqlite_database_path(tracking_uri)
    if database_path is None:
        return tracking_uri

    repo_root = repo_root.resolve()
    working_database_path = database_path

    if database_path.exists() and not _path_is_writable(database_path):
        working_database_path = resolve_runtime_write_path(repo_root, database_path)
        if not working_database_path.exists():
            working_database_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(database_path, working_database_path)

    if working_database_path.exists():
        _rewrite_tracking_store_paths(repo_root, working_database_path)

    return f"sqlite:///{working_database_path.as_posix()}"


def resolve_runtime_write_path(repo_root: Path, raw_path: str | Path) -> Path:
    # Prefer the original path, but fall back to .runtime when it is not writable.
    repo_root = repo_root.resolve()
    path = Path(raw_path)
    resolved_path = path if path.is_absolute() else (repo_root / path).resolve()
    if _path_is_writable(resolved_path):
        return resolved_path

    try:
        relative_path = resolved_path.relative_to(repo_root)
    except ValueError:
        if path.is_absolute():
            return resolved_path
        relative_path = path

    return (repo_root / RUNTIME_ROOT_RELATIVE_PATH / relative_path).resolve() 


def resolve_repo_bound_path(repo_root: Path, raw_path: str | Path) -> Path:
    # Rebind stored absolute paths to the current repo when possible.
    repo_root = repo_root.resolve()
    path = Path(raw_path)

    if not path.is_absolute():
        return (repo_root / path).resolve()

    if path.exists():
        return path

    previous_repo_root = _detect_previous_repo_root(path, repo_root.name)
    if previous_repo_root is None or previous_repo_root == repo_root:
        return path

    try:
        relative_path = path.relative_to(previous_repo_root)
    except ValueError:
        return path

    return (repo_root / relative_path).resolve()


def serialize_repo_bound_path(repo_root: Path, raw_path: str | Path) -> str:
    # Store repo-local paths relative to the current workspace root.
    resolved_path = resolve_repo_bound_path(repo_root, raw_path)
    try:
        return str(resolved_path.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved_path)


def resolve_runtime_model_artifact_path(repo_root: Path, model_source: str) -> Path | None:
    # Map a registry model id to its local copied artifact folder.
    if not model_source.startswith("models:/m-"):
        return None

    model_id = model_source.removeprefix("models:/")
    artifact_path = resolve_runtime_write_path(repo_root, Path("artifacts") / "models" / model_id / "artifacts")
    if not artifact_path.exists():
        return None
    return artifact_path


def _rewrite_tracking_store_paths(repo_root: Path, database_path: Path) -> int:
    # Rewrite stored MLflow paths so they point at the current repo copy.
    updated_rows = 0

    with sqlite3.connect(database_path) as connection:
        for table_name, column_name in TRACKING_PATH_COLUMNS:
            updated_rows += _rewrite_table_column(
                connection=connection,
                repo_root=repo_root,
                table_name=table_name,
                column_name=column_name,
            )
        if updated_rows:
            connection.commit()

    return updated_rows


def _resolve_sqlite_database_path(tracking_uri: str) -> Path | None:
    if tracking_uri.startswith("sqlite:////"):
        return Path(tracking_uri.removeprefix("sqlite:///"))

    if tracking_uri.startswith("sqlite:///"):
        return Path(tracking_uri.removeprefix("sqlite:///"))

    return None


def _rewrite_table_column(
    connection: sqlite3.Connection,
    repo_root: Path,
    table_name: str,
    column_name: str,
) -> int:
    # Update one MLflow table column when it contains stale paths.
    try:
        rows = connection.execute(f"SELECT rowid, {column_name} FROM {table_name}").fetchall()
    except sqlite3.Error:
        return 0

    updated_rows = 0
    for rowid, raw_value in rows:
        if raw_value is None:
            continue

        rewritten_value = _rewrite_tracking_value(
            repo_root=repo_root,
            table_name=table_name,
            column_name=column_name,
            raw_value=str(raw_value),
        )
        if rewritten_value == raw_value:
            continue

        connection.execute(
            f"UPDATE {table_name} SET {column_name} = ? WHERE rowid = ?",
            (rewritten_value, rowid),
        )
        updated_rows += 1

    return updated_rows


def _rewrite_tracking_value(
    repo_root: Path,
    table_name: str,
    column_name: str,
    raw_value: str,
) -> str:
    # Choose the right rewrite rule for each tracked MLflow path field.
    runtime_root = (repo_root / RUNTIME_ROOT_RELATIVE_PATH).resolve()

    if (table_name, column_name) == ("experiments", "artifact_location"):
        return _rewrite_repo_bound_value(raw_value, repo_root, runtime_root)

    if (table_name, column_name) in {
        ("logged_models", "artifact_location"),
        ("model_versions", "storage_location"),
    }:
        return _prepare_runtime_model_artifact(repo_root, raw_value)

    return _rewrite_repo_bound_value(raw_value, repo_root, repo_root)


def _rewrite_repo_bound_value(raw_value: str, repo_root: Path, target_root: Path) -> str:
    # Replace an old repo root in a stored path with the current repo root.
    extracted_path = _extract_filesystem_path(raw_value)
    if extracted_path is None:
        return raw_value

    previous_repo_root = _detect_previous_repo_root(extracted_path, repo_root.name)
    if previous_repo_root is None or previous_repo_root == repo_root:
        return raw_value

    try:
        relative_path = extracted_path.relative_to(previous_repo_root)
    except ValueError:
        return raw_value

    rewritten_path = target_root.resolve() / relative_path
    if raw_value.startswith("file://"):
        return rewritten_path.as_uri()
    return rewritten_path.as_posix()


def _prepare_runtime_model_artifact(repo_root: Path, raw_value: str) -> str:
    # Copy model artifacts to a writable location before serving them.
    resolved_value = _rewrite_repo_bound_value(raw_value, repo_root, repo_root)
    source_path = _extract_filesystem_path(resolved_value)
    if source_path is None or not source_path.exists():
        return resolved_value

    runtime_path = resolve_runtime_write_path(repo_root, source_path)
    if not runtime_path.exists():
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, runtime_path, dirs_exist_ok=True)

    _rewrite_model_metadata(runtime_path)
    if raw_value.startswith("file://"):
        return runtime_path.as_uri()
    return runtime_path.as_posix()


def _rewrite_model_metadata(artifact_path: Path) -> None:
    # Update the MLmodel file so its artifact path matches the copied location.
    mlmodel_path = artifact_path / "MLmodel"
    if not mlmodel_path.exists():
        return

    original_text = mlmodel_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    if not lines or not lines[0].startswith("artifact_path: "):
        return

    lines[0] = f"artifact_path: {artifact_path.as_uri()}"
    rewritten_text = "\n".join(lines)
    if original_text.endswith("\n"):
        rewritten_text += "\n"
    mlmodel_path.write_text(rewritten_text, encoding="utf-8")


def _extract_filesystem_path(raw_value: str) -> Path | None:
    # Parse filesystem paths from file URIs or plain absolute paths.
    if raw_value.startswith("file://"):
        parsed = urlparse(raw_value)
        return Path(unquote(parsed.path))

    if raw_value.startswith("/"):
        return Path(raw_value)

    return None


def _detect_previous_repo_root(path: Path, repo_name: str) -> Path | None:
    # Find the last matching repo directory inside an old absolute path.
    matching_indexes = [index for index, part in enumerate(path.parts) if part == repo_name]
    if not matching_indexes:
        return None

    repo_index = matching_indexes[-1]
    return _path_from_parts(path.parts[: repo_index + 1])


def _path_is_writable(path: Path) -> bool:
    # Check writability against the path itself or its nearest existing parent.
    existing_path = path if path.exists() else _nearest_existing_parent(path)
    return os.access(existing_path, os.W_OK)


def _nearest_existing_parent(path: Path) -> Path:
    # Walk up until a real parent path exists.
    current_path = path
    while not current_path.exists() and current_path != current_path.parent:
        current_path = current_path.parent
    return current_path


def _path_from_parts(parts: tuple[str, ...]) -> Path:
    path = Path(parts[0])
    for part in parts[1:]:
        path /= part
    return path