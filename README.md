# MLOps MNIST Drift Lab

This repository contains a lightweight end-to-end MLOps project built around MNIST digit classification. The project is intentionally scoped to emphasize the ML lifecycle rather than model complexity: reproducible data preparation, experiment tracking, model registration, deployment, monitoring, retraining, and safe rejection of degraded challengers.

## Scope

The MVP demonstrates three controlled production scenarios:

- successful retraining after receiving additional clean labeled data
- inference-time covariate drift through image corruption such as noise, blur, rotation, or contrast changes
- failed retraining caused by noisy labels, where the challenger model is rejected and the current production model remains active

## Stack

- TensorFlow and Keras for the digit classifier
- DVC for dataset and pipeline versioning
- MLflow for experiment tracking and model registry
- FastAPI for online inference
- Streamlit for a simple demo interface
- Prometheus and Grafana for monitoring
- Docker Compose as the MVP deployment target

## Repository Layout

```text
src/mlops_mnist/      Python package for data, training, API, monitoring, and utilities
configs/              Static configuration files beyond params.yaml
data/                 Local dataset structure for raw, interim, and processed artifacts
monitoring/           Prometheus and Grafana configuration assets
scripts/              One-off entrypoints and helper scripts
tests/                Smoke and unit tests
params.yaml           Project-level configuration for data scenarios and thresholds
```

## Day 1 Decisions

- Environment management uses `.venv` plus `requirements.txt`.
- Docker Compose is the only deployment target for the MVP.
- DVC remains the primary data versioning layer; MLflow handles experiments and model lifecycle.
- The initial data plan uses a small clean baseline set plus separate clean, drifted, and noisy future batches.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Next Steps

Day 2 work will implement the first executable slice of the pipeline:

- MNIST download and preprocessing
- deterministic scenario generation for clean incremental batches
- deterministic drifted inference batches
- deterministic noisy-label retraining batches