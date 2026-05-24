from __future__ import annotations

from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_module_from_source(module_name: str, module_path: Path):
    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    code = compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec")
    exec(code, module.__dict__)
    return module


def _bootstrap_api_module():
    _load_module_from_source(
        "mlops_mnist.utils.mlflow_compat",
        SRC_ROOT / "mlops_mnist" / "utils" / "mlflow_compat.py",
    )
    return _load_module_from_source(
        "mlops_mnist.api.app",
        SRC_ROOT / "mlops_mnist" / "api" / "app.py",
    )


main = _bootstrap_api_module().main


if __name__ == "__main__":
    raise SystemExit(main())