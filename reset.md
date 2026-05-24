cd /home/lsmliborio/projects/github/mlops/mlops-project && ./.venv/bin/python - <<'PY'
from pathlib import Path
import sys
repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "src"))
from mlops_mnist.app_reset import reset_application_state
print(reset_application_state(repo_root, params_path="params.yaml"))
PY