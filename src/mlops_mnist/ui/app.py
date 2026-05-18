from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image
from dotenv import dotenv_values


REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_BATCH_OPTIONS = [
    "baseline_test",
    "clean_batch_a",
    "clean_batch_b",
    "drifted_inference_batch",
    "noisy_retraining_batch",
]


def main() -> None:
    st.set_page_config(page_title="MNIST Drift Lab", page_icon="/", layout="wide")
    repo_root = Path(__file__).resolve().parents[3]
    runtime_config = _load_runtime_config(repo_root)
    api_base_url = st.sidebar.text_input(
        "API base URL",
        value=_default_api_base_url(runtime_config),
        help="Point this to the FastAPI inference service.",
    )

    st.title("MNIST Production Demo")
    st.caption("Predict with the current production alias and simulate small batch requests.")

    health = _get_api_health(api_base_url)
    _render_api_status(health)

    single_tab, upload_tab, batch_tab = st.tabs(
        ["Sample Prediction", "Upload Image", "Batch Simulation"]
    )

    with single_tab:
        _render_sample_prediction(repo_root, api_base_url)

    with upload_tab:
        _render_upload_prediction(api_base_url)

    with batch_tab:
        _render_batch_simulation(repo_root, api_base_url)


def _render_api_status(health: dict[str, Any] | None) -> None:
    if health is None:
        st.error("API is not reachable. Start the FastAPI service first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", health.get("status", "unknown"))
    col2.metric("Model", health.get("model_name", "-"))
    col3.metric("Alias", health.get("model_alias", "-"))
    col4.metric("Version", health.get("model_version", "-"))
    st.caption(
        f"Loaded candidate: {health.get('source_candidate_name', 'unknown')} | "
        f"Training run: {health.get('source_training_run_id', 'unknown')}"
    )


def _render_sample_prediction(repo_root: Path, api_base_url: str) -> None:
    st.subheader("Predict One Dataset Sample")
    batch_name = st.selectbox("Source batch", options=DEFAULT_BATCH_OPTIONS, key="sample_batch")
    batch = _load_batch(repo_root, batch_name)
    max_index = max(0, len(batch["images"]) - 1)
    sample_index = st.slider("Sample index", min_value=0, max_value=max_index, value=0, key="sample_index")

    image = batch["images"][sample_index, ..., 0]
    true_label = int(batch["labels"][sample_index])

    preview_col, result_col = st.columns([1, 1])
    with preview_col:
        st.image(image, clamp=True, width=220)
        st.write(f"True label: `{true_label}`")
        if batch.get("has_clean_labels"):
            st.caption("This batch also contains clean labels in the NPZ file.")

    with result_col:
        if st.button("Predict selected sample", key="predict_sample"):
            prediction = _predict_instances(api_base_url, [image], source=f"ui:{batch_name}")
            if prediction is None:
                st.error("Prediction request failed. Check that the API is running.")
            else:
                row = prediction["predictions"][0]
                st.success(f"Predicted label: {row['predicted_label']}")
                st.write(f"Confidence: `{row['confidence']:.4f}`")
                st.write(f"Model version: `{prediction['model_version']}`")


def _render_upload_prediction(api_base_url: str) -> None:
    st.subheader("Predict Uploaded Image")
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"], key="upload")
    if uploaded_file is None:
        st.caption("Upload a grayscale or RGB digit image. It will be resized to 28x28.")
        return

    image = _prepare_uploaded_image(uploaded_file.getvalue())
    preview_col, result_col = st.columns([1, 1])
    with preview_col:
        st.image(image, clamp=True, width=220)
    with result_col:
        if st.button("Predict uploaded image", key="predict_upload"):
            prediction = _predict_instances(api_base_url, [image], source="ui:upload")
            if prediction is None:
                st.error("Prediction request failed. Check that the API is running.")
            else:
                row = prediction["predictions"][0]
                st.success(f"Predicted label: {row['predicted_label']}")
                st.write(f"Confidence: `{row['confidence']:.4f}`")
                st.write(f"Model version: `{prediction['model_version']}`")


def _render_batch_simulation(repo_root: Path, api_base_url: str) -> None:
    st.subheader("Batch Simulation")
    batch_name = st.selectbox("Batch to send", options=DEFAULT_BATCH_OPTIONS, key="batch_simulation_name")
    batch_size = st.slider("Batch size", min_value=1, max_value=64, value=16, key="batch_size")
    batch = _load_batch(repo_root, batch_name)
    images = batch["images"][:batch_size, ..., 0]
    labels = batch["labels"][:batch_size]

    if st.button("Run batch prediction", key="predict_batch"):
        prediction = _predict_instances(api_base_url, list(images), source=f"ui-batch:{batch_name}")
        if prediction is None:
            st.error("Batch prediction failed. Check that the API is running.")
            return

        predicted_labels = [row["predicted_label"] for row in prediction["predictions"]]
        confidences = [row["confidence"] for row in prediction["predictions"]]
        accuracy = float(np.mean(np.asarray(predicted_labels) == labels)) if len(labels) else 0.0

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Batch accuracy", f"{accuracy:.4f}")
        metric_col2.metric("Mean confidence", f"{np.mean(confidences):.4f}")
        metric_col3.metric("Model version", prediction["model_version"])

        results = []
        for index, predicted_label in enumerate(predicted_labels):
            results.append(
                {
                    "index": index,
                    "true_label": int(labels[index]),
                    "predicted_label": int(predicted_label),
                    "confidence": float(confidences[index]),
                }
            )
        st.dataframe(results, use_container_width=True)


def _load_runtime_config(repo_root: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    example_path = repo_root / ".env.example"
    env_path = repo_root / ".env"

    if example_path.exists():
        config.update({key: value for key, value in dotenv_values(example_path).items() if value})
    if env_path.exists():
        config.update({key: value for key, value in dotenv_values(env_path).items() if value})

    return config


def _default_api_base_url(runtime_config: dict[str, str]) -> str:
    api_port = runtime_config.get("API_PORT", "8000")
    return f"http://127.0.0.1:{api_port}"


@st.cache_data(show_spinner=False)
def _load_batch(repo_root: Path, batch_name: str) -> dict[str, Any]:
    batch_path = repo_root / "data" / "processed" / f"{batch_name}.npz"
    with np.load(batch_path) as batch:
        payload = {
            "images": batch["images"].astype(np.float32),
            "labels": batch["labels"].astype(np.int64),
            "has_clean_labels": "clean_labels" in batch.files,
        }
    return payload


def _get_api_health(api_base_url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _predict_instances(
    api_base_url: str,
    instances: list[np.ndarray] | list[list[list[float]]],
    source: str,
) -> dict[str, Any] | None:
    payload_instances = []
    for image in instances:
        if isinstance(image, np.ndarray):
            payload_instances.append(image.tolist())
        else:
            payload_instances.append(image)

    try:
        response = requests.post(
            f"{api_base_url}/predict",
            json={"instances": payload_instances, "source": source},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _prepare_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


if __name__ == "__main__":
    main()