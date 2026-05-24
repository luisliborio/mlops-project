"""Streamlit demo UI package."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["main"]


def __getattr__(name: str) -> Any:
	if name == "main":
		module = import_module(".app", __name__)
		return getattr(module, name)

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")