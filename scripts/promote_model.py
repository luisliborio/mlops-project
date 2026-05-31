from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Allow this wrapper script to import the local package.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_mnist.training.registry import main


if __name__ == "__main__":
    # Forward CLI arguments to the promotion entrypoint.
    raise SystemExit(main())