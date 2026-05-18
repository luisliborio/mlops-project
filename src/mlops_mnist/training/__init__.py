"""Training and evaluation package."""

from .registry import promote_best_model
from .trainer import build_cnn_model, train_candidates

__all__ = ["build_cnn_model", "promote_best_model", "train_candidates"]
