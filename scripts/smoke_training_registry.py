from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Allow this wrapper script to import the local package.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_mnist.training.evaluation import run_training_registry_smoke


def main() -> int:
    # Run the small registry smoke check and print its summary.
    summary = run_training_registry_smoke()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    # Exit with the smoke check status code.
    raise SystemExit(main())