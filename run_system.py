from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_mnist.utils.runtime import load_runtime_config, resolve_runtime_port, resolve_tracking_uri


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the local MNIST API, Streamlit UI, and MLflow UI together.",
    )
    parser.add_argument("--api-host", default=None, help="Optional API host override.")
    parser.add_argument("--api-port", type=int, default=None, help="Optional API port override.")
    parser.add_argument("--ui-port", type=int, default=None, help="Optional Streamlit port override.")
    parser.add_argument("--mlflow-port", type=int, default=None, help="Optional MLflow UI port override.")
    args = parser.parse_args(argv)
    runtime_config = load_runtime_config(REPO_ROOT)

    api_command = [sys.executable, "scripts/run_api.py"]
    if args.api_host is not None:
        api_command.extend(["--host", args.api_host])
    if args.api_port is not None:
        api_command.extend(["--port", str(args.api_port)])

    ui_command = [sys.executable, "-m", "streamlit", "run", "scripts/run_ui.py"]
    if args.ui_port is not None:
        ui_command.extend(["--server.port", str(args.ui_port)])

    mlflow_port = args.mlflow_port or resolve_runtime_port(runtime_config, "MLFLOW_UI_PORT", 5000)
    tracking_uri = resolve_tracking_uri(
        REPO_ROOT,
        runtime_config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )

    process_specs = [
        ("API", api_command),
        ("UI", ui_command),
    ]
    if _port_is_open("127.0.0.1", mlflow_port):
        print(f"MLflow UI port {mlflow_port} is already in use. Leaving the existing service untouched.")
    else:
        process_specs.append(
            (
                "MLflow UI",
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "ui",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(mlflow_port),
                    "--workers",
                    "1",
                    "--backend-store-uri",
                    tracking_uri,
                ],
            )
        )

    processes = [subprocess.Popen(command, cwd=REPO_ROOT) for _, command in process_specs]

    print("Started the local MLOps services. Press Ctrl+C to stop them.")
    print(f"API command: {' '.join(api_command)}")
    print(f"UI command: {' '.join(ui_command)}")
    if len(process_specs) > 2:
        print(f"MLflow UI command: {' '.join(process_specs[2][1])}")
    print(f"MLflow UI URL: http://127.0.0.1:{mlflow_port}")

    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return _shutdown_processes(processes, return_code)
            time.sleep(1)
    except KeyboardInterrupt:
        return _shutdown_processes(processes, 0)


def _shutdown_processes(processes: list[subprocess.Popen[bytes]], exit_code: int) -> int:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    return exit_code


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


if __name__ == "__main__":
    raise SystemExit(main())