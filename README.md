# MLOps MNIST Drift Lab

This project is a small end-to-end MLOps demo built on MNIST. It prepares data with DVC, tracks and registers models in MLflow, serves predictions through FastAPI, exposes a Streamlit control UI, and shows monitoring in Prometheus and Grafana.

The app demonstrates three behaviors:

- normal production inference
- drift detection on corrupted incoming batches
- retraining with acceptance or rejection of a challenger model

There are two main dashboards under Grafana's umbrella:

- Unsupervised metrics; tracks label-agnostic parameters 
- Supervised metrics; tracks traditional metrics that are available after the system records labels

## Recommended Run Path

Use Docker Compose.

Requirements:

- Docker
- Docker Compose
- Used Local ports: `8000`, `8501`, `5000`, `9090`, `3000`

From the repository root:

```bash
docker compose up -d --build
```

Streamlit UI (Simulation Control Panel):

- `http://127.0.0.1:8501`

Other Endpoints:

- API ----------- `http://127.0.0.1:8000`
- MLflow: ------ `http://127.0.0.1:5000`
- Prometheus: - `http://127.0.0.1:9090`
- Grafana: ------ `http://127.0.0.1:3000`

## First-Time Setup In The UI

After the containers are running:

1. open the `Initial Setup` page in Streamlit
2. click `Run dvc repro` (See `dvc.yaml` stages)
3. click `Run initial training + promote winner`

After that, the app is ready for predictions, lifecycle simulation, MLflow inspection, and monitoring.

To stop the stack:

```bash
docker compose down
```

## Local Run Without Docker

Use this only if you want to run the API, UI, and MLflow UI directly from the Python environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
.venv/bin/python run_system.py
```

This starts:

- FastAPI
- Streamlit
- MLflow UI

If you also want Prometheus and Grafana, use Docker Compose.

## If You Change `params.yaml`

Restarting containers is only enough for settings that are read directly at app startup.

- If you changed `monitoring.*` or `retraining.*`, restart the relevant services.
- If you changed `data.*` or `model.*`, restart is not enough.

For `data.*` or `model.*` changes, use this flow:

1. open the `Initial Setup` page
2. click `Hard reset to empty state`
3. click `Run dvc repro`
4. click `Run initial training + promote winner`

This is necessary because those settings affect generated datasets, frozen baseline artifacts, and trained model outputs.

## Generated State

The Docker setup bind-mounts the repository into the containers. Generated files stay in the project folder, including:

- `mlflow.db`
- `artifacts/`
- `data/`
- `logs/`
- `models/`
- `ui_lifecycle/`