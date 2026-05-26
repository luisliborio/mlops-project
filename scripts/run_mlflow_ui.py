from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_mnist.utils.runtime import load_runtime_config, prepare_runtime_tracking_uri, resolve_runtime_port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local MLflow UI with repo-aware tracking paths.")
    parser.add_argument("--host", default=None, help="Optional host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional port override.")
    parser.add_argument("--workers", type=int, default=1, help="MLflow UI worker count.")
    args = parser.parse_args(argv)

    runtime_config = load_runtime_config(REPO_ROOT)
    tracking_uri = prepare_runtime_tracking_uri(REPO_ROOT, runtime_config)
    host = args.host or "0.0.0.0"
    port = args.port or resolve_runtime_port(runtime_config, "MLFLOW_UI_PORT", 5000)

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(args.workers),
        "--backend-store-uri",
        tracking_uri,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    raise SystemExit(main())