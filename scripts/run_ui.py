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


def _bootstrap_ui_module():
    module_paths = [
        ("mlops_mnist.utils.mlflow_compat", SRC_ROOT / "mlops_mnist" / "utils" / "mlflow_compat.py"),
        ("mlops_mnist.training.trainer", SRC_ROOT / "mlops_mnist" / "training" / "trainer.py"),
        ("mlops_mnist.training.registry", SRC_ROOT / "mlops_mnist" / "training" / "registry.py"),
        ("mlops_mnist.training.evaluation", SRC_ROOT / "mlops_mnist" / "training" / "evaluation.py"),
        ("mlops_mnist.monitoring.analyzer", SRC_ROOT / "mlops_mnist" / "monitoring" / "analyzer.py"),
        ("mlops_mnist.training.retraining", SRC_ROOT / "mlops_mnist" / "training" / "retraining.py"),
        ("mlops_mnist.ui.lifecycle", SRC_ROOT / "mlops_mnist" / "ui" / "lifecycle.py"),
    ]
    for module_name, module_path in module_paths:
        _load_module_from_source(module_name, module_path)
    return _load_module_from_source(
        "mlops_mnist.ui.app",
        SRC_ROOT / "mlops_mnist" / "ui" / "app.py",
    )


main = _bootstrap_ui_module().main


main()