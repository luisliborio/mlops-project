"""Training and evaluation package."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
	"build_cnn_model",
	"decide_challenger_promotion",
	"evaluate_registered_model",
	"promote_best_model",
	"run_retraining_workflow",
	"run_training_registry_smoke",
	"train_candidates",
]


def __getattr__(name: str) -> Any:
	if name in {"build_cnn_model", "train_candidates"}:
		module = import_module(".trainer", __name__)
		return getattr(module, name)

	if name in {"evaluate_registered_model", "run_training_registry_smoke"}:
		module = import_module(".evaluation", __name__)
		return getattr(module, name)

	if name == "promote_best_model":
		module = import_module(".registry", __name__)
		return getattr(module, name)

	if name in {"decide_challenger_promotion", "run_retraining_workflow"}:
		module = import_module(".retraining", __name__)
		return getattr(module, name)

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
