from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SETUP_STATUS_RELATIVE_PATH = Path("artifacts/ui_setup/status.json")
TRAINING_SUMMARY_GLOB = "artifacts/training/*/summary.json"
PROMOTION_SUMMARY_GLOB = "artifacts/registry/**/promotion_summary.json"
STATUS_HISTORY_LIMIT = 20
SETUP_MODE_EMPTY = "empty"
SETUP_MODE_INITIALIZED = "initialized"
SETUP_MODE_UNKNOWN = "unknown"


def load_setup_status(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    status_path = repo_root / SETUP_STATUS_RELATIVE_PATH
    if not status_path.exists():
        return {
            "mode": SETUP_MODE_UNKNOWN,
            "last_action": None,
            "history": [],
        }

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload.setdefault("mode", SETUP_MODE_UNKNOWN)
    payload.setdefault("last_action", None)
    payload.setdefault("history", [])
    return payload


def save_setup_status(repo_root: str | Path, payload: dict[str, Any]) -> Path:
    repo_root = Path(repo_root).resolve()
    status_path = repo_root / SETUP_STATUS_RELATIVE_PATH
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return status_path


def record_setup_action(
    repo_root: str | Path,
    *,
    action_name: str,
    outcome: str,
    message: str,
    command: list[str] | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = load_setup_status(repo_root)
    action = {
        "timestamp": _utc_now(),
        "name": action_name,
        "outcome": outcome,
        "message": message,
        "command": command or [],
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
        "metadata": metadata or {},
    }

    history = list(status.get("history", []))
    history.append(action)
    status["history"] = history[-STATUS_HISTORY_LIMIT:]
    status["last_action"] = action
    if mode is not None:
        status["mode"] = mode
    save_setup_status(repo_root, status)
    return action


def list_dvc_output_paths(repo_root: str | Path, dvc_path: str | Path = "dvc.yaml") -> list[Path]:
    repo_root = Path(repo_root).resolve()
    resolved_dvc_path = _resolve_path(repo_root, dvc_path)
    if not resolved_dvc_path.exists():
        return []

    payload = yaml.safe_load(resolved_dvc_path.read_text(encoding="utf-8")) or {}
    stages = payload.get("stages", {})
    output_paths: list[Path] = []
    for stage_config in stages.values():
        for output in stage_config.get("outs", []) or []:
            output_paths.append(Path(str(output)))

    return sorted(dict.fromkeys(output_paths))


def latest_training_summary(repo_root: str | Path) -> dict[str, Any] | None:
    summary_path = _latest_match(Path(repo_root).resolve(), TRAINING_SUMMARY_GLOB)
    if summary_path is None:
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    best_result = None
    if results:
        best_result = max(
            results,
            key=lambda item: (
                float(item.get("best_validation_accuracy", 0.0)),
                float(item.get("test_accuracy", 0.0)),
                float(item.get("validation_macro_f1", 0.0)),
            ),
        )

    return {
        "path": str(summary_path.relative_to(Path(repo_root).resolve())),
        "results": results,
        "candidate_names": [str(result.get("candidate_name", "unknown")) for result in results],
        "best_candidate": best_result.get("candidate_name") if best_result is not None else None,
        "results_count": len(results),
    }


def latest_promotion_summary(repo_root: str | Path) -> dict[str, Any] | None:
    summary_path = _latest_match(Path(repo_root).resolve(), PROMOTION_SUMMARY_GLOB)
    if summary_path is None:
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["path"] = str(summary_path.relative_to(Path(repo_root).resolve()))
    return payload


def evaluate_setup_state(repo_root: str | Path, dvc_path: str | Path = "dvc.yaml") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    status = load_setup_status(repo_root)
    dvc_outputs = list_dvc_output_paths(repo_root, dvc_path=dvc_path)
    missing_outputs = [
        str(output)
        for output in dvc_outputs
        if not (repo_root / output).exists()
    ]
    training = latest_training_summary(repo_root)
    promotion = latest_promotion_summary(repo_root)

    data_ready = bool(dvc_outputs) and not missing_outputs
    training_ready = training is not None and training["results_count"] > 0
    production_ready = promotion is not None
    setup_complete = data_ready and training_ready and production_ready

    mode = str(status.get("mode") or SETUP_MODE_UNKNOWN)
    if setup_complete:
        mode = SETUP_MODE_INITIALIZED
    elif mode not in {SETUP_MODE_EMPTY, SETUP_MODE_UNKNOWN, SETUP_MODE_INITIALIZED}:
        mode = SETUP_MODE_UNKNOWN

    return {
        "mode": mode,
        "data_ready": data_ready,
        "training_ready": training_ready,
        "production_ready": production_ready,
        "setup_complete": setup_complete,
        "data_outputs": [str(path) for path in dvc_outputs],
        "missing_data_outputs": missing_outputs,
        "training_summary": training,
        "promotion_summary": promotion,
        "last_action": status.get("last_action"),
        "history": list(status.get("history", [])),
    }


def resolve_dvc_command() -> list[str]:
    dvc_executable = shutil.which("dvc")
    if dvc_executable is not None:
        return [dvc_executable]
    return [sys.executable, "-m", "dvc"]


def resolve_python_command(script_path: str) -> list[str]:
    return [sys.executable, script_path]


def run_setup_command(
    repo_root: str | Path,
    *,
    command: list[str],
    action_name: str,
    success_message: str,
    failure_message: str,
    mode_on_success: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    message = success_message if ok else failure_message
    action = record_setup_action(
        repo_root,
        action_name=action_name,
        outcome="success" if ok else "error",
        message=message,
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        mode=mode_on_success if ok else None,
        metadata={
            "return_code": result.returncode,
            **(metadata or {}),
        },
    )
    return {
        "ok": ok,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "action": action,
    }


def _latest_match(repo_root: Path, pattern: str) -> Path | None:
    matches = list(repo_root.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _tail_text(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()