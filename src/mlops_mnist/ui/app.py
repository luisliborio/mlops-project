from __future__ import annotations

from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from dotenv import dotenv_values

from mlops_mnist.app_reset import reset_application_state, reset_to_pre_setup_state
from mlops_mnist.setup_state import (
    SETUP_MODE_INITIALIZED,
    evaluate_setup_state,
    latest_promotion_summary,
    latest_training_summary,
    resolve_dvc_command,
    resolve_python_command,
    run_setup_command,
)
from mlops_mnist.ui.lifecycle import (
    advance_stream_cursor,
    append_event_log,
    append_recorded_batch,
    clear_labeled_batches,
    compute_cursor_slice,
    compute_histogram_js_divergence,
    count_pending_labeled_samples,
    count_ready_labeled_samples,
    ensure_baseline_reference,
    latest_recorded_distribution,
    load_lifecycle_state,
    load_project_params,
    record_preview_batch,
    reset_lifecycle_state,
    run_lifecycle_monitoring,
    run_lifecycle_retraining,
    save_lifecycle_state,
    summarize_unsupervised_alerts,
    switch_lifecycle_batch,
)
from mlops_mnist.utils.mlflow_compat import resolve_repo_bound_path


REQUEST_TIMEOUT_SECONDS = 30
FLASH_MESSAGE_KEY = "ui_flash_message"
ACTIVE_VIEW_KEY = "ui_active_view"
PENDING_FULL_RESET_KEY = "ui_pending_full_system_reset"
INITIAL_SETUP_VIEW = "Initial Setup"
SAMPLE_PREDICTION_VIEW = "Sample Prediction"
UPLOAD_IMAGE_VIEW = "Upload Image"
LIFECYCLE_VIEW = "Lifecycle Simulation"
VIEW_OPTIONS = [INITIAL_SETUP_VIEW, SAMPLE_PREDICTION_VIEW, UPLOAD_IMAGE_VIEW, LIFECYCLE_VIEW]
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
    params = load_project_params(repo_root / "params.yaml")
    setup_state = evaluate_setup_state(repo_root)
    runtime_config = _load_runtime_config(repo_root)
    api_base_url = st.sidebar.text_input(
        "API base URL",
        value=_default_api_base_url(runtime_config),
        help="Point this to the FastAPI inference service.",
    )

    st.title("MNIST Production Demo")
    st.caption("Predict with the current production alias and simulate small batch requests.")
    _render_flash_message()

    if ACTIVE_VIEW_KEY not in st.session_state:
        st.session_state[ACTIVE_VIEW_KEY] = (
            INITIAL_SETUP_VIEW if not setup_state["setup_complete"] else SAMPLE_PREDICTION_VIEW
        )

    health = _get_api_health(api_base_url)
    _render_api_status(health)

    active_view = st.radio(
        "View",
        options=VIEW_OPTIONS,
        key=ACTIVE_VIEW_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_view == INITIAL_SETUP_VIEW:
        _render_initial_setup(repo_root, api_base_url, setup_state, health)

    elif not setup_state["setup_complete"]:
        _render_setup_required(setup_state)

    elif active_view == SAMPLE_PREDICTION_VIEW:
        _render_sample_prediction(repo_root, api_base_url)

    elif active_view == UPLOAD_IMAGE_VIEW:
        _render_upload_prediction(api_base_url)

    else:
        _render_lifecycle_simulation(repo_root, api_base_url, params)


def _render_api_status(health: dict[str, Any] | None) -> None:
    if health is None:
        st.warning("API is not reachable. Start the FastAPI service when you want live predictions.")
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
    if health.get("status") != "ok":
        st.warning("API is running but no production model is loaded yet. Complete Initial Setup or reload after a promotion.")


def _render_initial_setup(
    repo_root: Path,
    api_base_url: str,
    setup_state: dict[str, Any],
    health: dict[str, Any] | None,
) -> None:
    st.subheader("Initial Setup")
    st.caption(
        "Use this page for the cold project state before `dvc repro`. "
        "Run data preparation, launch the initial competitor training, promote the winner, and hard reset back to empty when needed."
    )

    if setup_state["setup_complete"]:
        st.success("Initial setup is complete. You can inspect the recorded outputs below, rerun the bootstrap flow, or hard reset back to the empty state.")
    elif setup_state.get("mode") == "empty":
        st.info("The repository is in the empty pre-setup state. Start with `dvc repro`.")
    else:
        st.info("Initial setup is incomplete. Run `dvc repro` first, then train and promote the initial winner.")

    training_summary = setup_state.get("training_summary") or {}
    promotion_summary = setup_state.get("promotion_summary") or {}
    production_version = None
    if health is not None and health.get("status") == "ok":
        production_version = health.get("model_version")
    elif promotion_summary:
        production_version = promotion_summary.get("registered_version")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("DVC data", "ready" if setup_state["data_ready"] else "pending")
    status_col2.metric("Training", "ready" if setup_state["training_ready"] else "pending")
    status_col3.metric("Production", f"v{production_version}" if production_version is not None else "pending")
    status_col4.metric(
        "API",
        "ok" if health is not None and health.get("status") == "ok" else ("unavailable" if health is not None else "offline"),
    )

    if training_summary:
        st.caption(
            "Latest training summary: "
            f"{training_summary['path']} | candidates: {', '.join(training_summary['candidate_names'])} | best: {training_summary.get('best_candidate') or 'unknown'}"
        )
    if promotion_summary:
        st.caption(
            "Latest promotion summary: "
            f"{promotion_summary['path']} | version: {promotion_summary.get('registered_version', 'unknown')}"
        )

    last_action = setup_state.get("last_action")
    if last_action is not None:
        if last_action.get("outcome") == "error":
            st.warning(last_action.get("message", "The last setup action failed."))
        else:
            st.caption(f"Last action: {last_action.get('message', '')}")

        command = last_action.get("command") or []
        if command:
            st.caption("Last command: " + " ".join(command))

        stdout_tail = last_action.get("stdout_tail") or ""
        stderr_tail = last_action.get("stderr_tail") or ""
        if stdout_tail or stderr_tail:
            with st.expander("Latest setup command output"):
                if stdout_tail:
                    st.code(stdout_tail)
                if stderr_tail:
                    st.code(stderr_tail)

    if setup_state.get("missing_data_outputs"):
        missing_preview = ", ".join(setup_state["missing_data_outputs"][:3])
        suffix = "" if len(setup_state["missing_data_outputs"]) <= 3 else ", ..."
        st.caption(f"Missing DVC outputs: {missing_preview}{suffix}")

    control_col1, control_col2, control_col3 = st.columns(3)
    run_dvc_clicked = control_col1.button("Run dvc repro", use_container_width=True)
    train_clicked = control_col2.button(
        "Run initial training + promote winner",
        use_container_width=True,
        disabled=not setup_state["data_ready"],
    )
    hard_reset_clicked = control_col3.button("Hard reset to empty state", use_container_width=True)

    if run_dvc_clicked:
        _handle_initial_dvc_repro(repo_root)

    if train_clicked:
        _handle_initial_training_and_promotion(repo_root, api_base_url)

    if hard_reset_clicked:
        _handle_pre_setup_reset(repo_root, api_base_url)


def _render_setup_required(setup_state: dict[str, Any]) -> None:
    st.subheader("Setup Required")
    st.info("Complete Initial Setup before using the prediction and lifecycle views.")
    if setup_state.get("missing_data_outputs"):
        missing_preview = ", ".join(setup_state["missing_data_outputs"][:5])
        suffix = "" if len(setup_state["missing_data_outputs"]) <= 5 else ", ..."
        st.caption(f"Missing outputs: {missing_preview}{suffix}")
    else:
        st.caption("The current project state does not yet expose the baseline data, training outputs, and promoted production model required by the other pages.")


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


def _render_lifecycle_simulation(repo_root: Path, api_base_url: str, params: dict[str, Any]) -> None:
    st.subheader("Lifecycle Simulation")
    st.caption(
        "Each click is one simulated time step. Record off means preview only. "
        "Record on means incoming data is stored for monitoring, labeling, and retraining."
    )

    _consume_pending_full_system_reset(repo_root=repo_root, api_base_url=api_base_url)
    state = load_lifecycle_state(repo_root)
    default_batch = (
        state.get("active_batch_name")
        if state.get("active_batch_name") in DEFAULT_BATCH_OPTIONS
        else DEFAULT_BATCH_OPTIONS[0]
    )
    batch_selectbox_kwargs: dict[str, Any] = {
        "label": "Incoming source batch",
        "options": DEFAULT_BATCH_OPTIONS,
        "key": "lifecycle_batch_name",
    }
    if "lifecycle_batch_name" not in st.session_state:
        batch_selectbox_kwargs["index"] = DEFAULT_BATCH_OPTIONS.index(default_batch)
    batch_name = st.selectbox(**batch_selectbox_kwargs)

    batch_slider_kwargs: dict[str, Any] = {
        "label": "Incoming batch size",
        "min_value": 1,
        "max_value": 512,
        "key": "lifecycle_batch_size",
    }
    if "lifecycle_batch_size" not in st.session_state:
        batch_slider_kwargs["value"] = int(state.get("batch_size", 16) or 16)
    batch_size = st.slider(**batch_slider_kwargs)
    batch = _load_batch(repo_root, batch_name)

    if batch_name != state.get("active_batch_name"):
        if not state.get("baseline_reference"):
            baseline_reference = ensure_baseline_reference(
                repo_root=repo_root,
                params_path=repo_root / "params.yaml",
                baseline_snapshot_path=state.get("baseline_snapshot_path"),
            )
            state["baseline_snapshot_path"] = baseline_reference["summary_path"]
            state["baseline_reference"] = baseline_reference
        state = switch_lifecycle_batch(
            repo_root=repo_root,
            state=state,
            batch_name=batch_name,
            batch_size=batch_size,
        )
        st.info(
            f"Incoming source batch switched to {batch_name}. "
            "Lifecycle history and recorded data were preserved."
        )
    elif not state.get("baseline_reference"):
        baseline_reference = ensure_baseline_reference(
            repo_root=repo_root,
            params_path=repo_root / "params.yaml",
            baseline_snapshot_path=state.get("baseline_snapshot_path"),
        )
        state["baseline_snapshot_path"] = baseline_reference["summary_path"]
        state["baseline_reference"] = baseline_reference
        save_lifecycle_state(repo_root, state)

    record_mode = st.toggle(
        "Record incoming data",
        value=bool(state.get("record_mode", False)),
        help="When on, each batch is stored as recent incoming data for lifecycle monitoring.",
    )
    if bool(record_mode) != bool(state.get("record_mode", False)):
        state["record_mode"] = bool(record_mode)
        save_lifecycle_state(repo_root, state)

    min_recent_samples = int(params["retraining"]["min_recent_samples"])
    _render_lifecycle_status_cards(state, batch_name, batch_size, len(batch["images"]), min_recent_samples)

    next_slice = compute_cursor_slice(
        cursor=int(state.get("cursor", 0)),
        batch_size=int(batch_size),
        total_examples=len(batch["images"]),
    )
    pending_labeled = count_pending_labeled_samples(state)
    ready_labeled = count_ready_labeled_samples(state)

    control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns(5)
    run_next_clicked = control_col1.button(
        "Run Next Batch",
        use_container_width=True,
        disabled=not bool(next_slice["indices"]),
    )
    label_clicked = control_col2.button(
        "Label Data",
        use_container_width=True,
        disabled=pending_labeled <= 0,
    )
    clear_labeled_clicked = control_col3.button(
        "Clean Labeled Data",
        use_container_width=True,
        disabled=ready_labeled <= 0,
    )
    retrain_clicked = control_col4.button(
        "Retrain",
        use_container_width=True,
        disabled=ready_labeled < min_recent_samples,
    )
    control_col5.button(
        "Full System Reset",
        use_container_width=True,
        on_click=_queue_full_system_reset,
    )

    if run_next_clicked:
        _handle_run_next_batch(
            repo_root=repo_root,
            api_base_url=api_base_url,
            params=params,
            state=state,
            batch_name=batch_name,
            batch_size=batch_size,
            batch=batch,
            record_mode=bool(record_mode),
        )

    if label_clicked:
        _handle_label_action(repo_root=repo_root, state=state, params=params)

    if clear_labeled_clicked:
        _handle_clear_labeled_action(repo_root=repo_root, state=state)

    if retrain_clicked:
        _handle_retraining_action(
            repo_root=repo_root,
            api_base_url=api_base_url,
            state=state,
            params=params,
        )

    _render_latest_preview(state.get("latest_preview"))
    _render_alert_panel(state, params)
    _render_placeholder_dashboards(state)
    _render_event_log(state)


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


def _reload_api_model(api_base_url: str) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{api_base_url}/reload-model", timeout=REQUEST_TIMEOUT_SECONDS)
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


def _render_lifecycle_status_cards(
    state: dict[str, Any],
    batch_name: str,
    batch_size: int,
    total_examples: int,
    min_recent_samples: int,
) -> None:
    next_slice = compute_cursor_slice(
        cursor=int(state.get("cursor", 0)),
        batch_size=int(batch_size),
        total_examples=int(total_examples),
    )
    next_range = (
        f"{next_slice['start_index']}-{next_slice['end_index'] - 1}"
        if next_slice["indices"]
        else "end reached"
    )
    pending_labeled = count_pending_labeled_samples(state)
    ready_labeled = count_ready_labeled_samples(state)
    current_alerts = state.get("current_alerts", [])
    baseline_reference = state.get("baseline_reference") or {}
    training_reference_examples = baseline_reference.get("reference_data_examples")
    training_reference_name = baseline_reference.get("reference_data_name")
    benchmark_accuracy = baseline_reference.get("accuracy")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    col1.metric("Active stream", batch_name)
    col2.metric("Cursor", f"{state.get('cursor', 0)} / {total_examples}")
    col3.metric("Next slice", next_range)
    col4.metric("Record", "on" if state.get("record_mode") else "off")
    col5.metric("Unlabeled", str(pending_labeled))
    col6.metric("Labeled / retrain", f"{ready_labeled} / {min_recent_samples}")
    col7.metric(
        "Benchmark acc",
        f"{float(benchmark_accuracy):.4f}" if benchmark_accuracy is not None else "-",
    )
    col8.metric(
        "Train ref size",
        str(training_reference_examples) if training_reference_examples is not None else "-",
    )

    if training_reference_name:
        st.caption(f"Current production training reference: {training_reference_name}")

    if current_alerts:
        st.warning("Active alerts: " + ", ".join(current_alerts))
    else:
        st.success("No active lifecycle alerts.")


def _handle_run_next_batch(
    repo_root: Path,
    api_base_url: str,
    params: dict[str, Any],
    state: dict[str, Any],
    batch_name: str,
    batch_size: int,
    batch: dict[str, Any],
    record_mode: bool,
) -> None:
    previous_cursor = int(state.get("cursor", 0))
    previous_step = int(state.get("step_counter", 0))
    previous_exhausted = bool(state.get("stream_exhausted", False))

    slice_payload = advance_stream_cursor(state, total_examples=len(batch["images"]), batch_size=batch_size)
    if not slice_payload["indices"]:
        append_event_log(state, level="warning", message="The selected stream has reached its end. Reset the scenario to start again.")
        save_lifecycle_state(repo_root, state)
        st.warning("The selected stream has reached its end. Click Reset Scenario to replay it.")
        return

    images = batch["images"][slice_payload["indices"], ..., 0]
    labels = batch["labels"][slice_payload["indices"]]
    current_images = batch["images"][slice_payload["indices"]]

    start_time = perf_counter()
    prediction = _predict_instances(api_base_url, list(images), source=f"ui-lifecycle:{batch_name}")
    latency_ms = (perf_counter() - start_time) * 1000.0
    if prediction is None:
        state["cursor"] = previous_cursor
        state["step_counter"] = previous_step
        state["stream_exhausted"] = previous_exhausted
        save_lifecycle_state(repo_root, state)
        st.error("The batch prediction failed, so the lifecycle cursor was not advanced.")
        return

    preview_payload = _build_preview_payload(
        slice_payload=slice_payload,
        labels=labels,
        prediction=prediction,
        latency_ms=latency_ms,
        labels_hidden=record_mode,
    )
    record_preview_batch(state, preview_payload)

    if record_mode:
        baseline_reference = state.get("baseline_reference") or ensure_baseline_reference(
            repo_root=repo_root,
            params_path=repo_root / "params.yaml",
            baseline_snapshot_path=state.get("baseline_snapshot_path"),
        )
        state["baseline_snapshot_path"] = baseline_reference["summary_path"]
        state["baseline_reference"] = baseline_reference
        baseline_images = _load_images_from_path(
            repo_root,
            baseline_reference.get("reference_data_path") or baseline_reference["batch_path"],
        )
        alert_summary = summarize_unsupervised_alerts(
            thresholds=params["monitoring"]["thresholds"],
            baseline_mean_confidence=float(baseline_reference["mean_confidence"]),
            current_mean_confidence=float(prediction["mean_confidence"]),
            histogram_js_divergence=compute_histogram_js_divergence(baseline_images, current_images),
        )
        append_recorded_batch(
            state=state,
            batch_name=batch_name,
            step=int(slice_payload["step"]),
            slice_payload=slice_payload,
            prediction=prediction,
            latency_ms=latency_ms,
            histogram_js_divergence=float(alert_summary["histogram_js_divergence"]),
            confidence_drop=float(alert_summary["confidence_drop"]),
            alerts=list(alert_summary["alerts"]),
            clean_labels_available=bool(batch.get("has_clean_labels", False)),
        )
        range_text = f"{slice_payload['start_index']}-{slice_payload['end_index'] - 1}"
        if alert_summary["alerts"]:
            append_event_log(
                state,
                level="warning",
                message=(
                    f"Recorded batch {range_text} from {batch_name}. Alerts raised: "
                    + ", ".join(alert_summary["alerts"])
                ),
            )
            _set_flash_message(
                "warning",
                "Recorded a new incoming batch and raised alerts: "
                + ", ".join(alert_summary["alerts"]),
            )
        else:
            append_event_log(
                state,
                level="info",
                message=f"Recorded batch {range_text} from {batch_name} without active alerts.",
            )
            _set_flash_message("success", f"Recorded batch {range_text} from {batch_name}.")
    else:
        append_event_log(
            state,
            level="info",
            message=(
                f"Previewed batch {slice_payload['start_index']}-{slice_payload['end_index'] - 1} "
                f"from {batch_name} without recording it."
            ),
        )
        _set_flash_message(
            "info",
            "Preview-only mode: this batch was inferred but not stored for monitoring.",
        )

    save_lifecycle_state(repo_root, state)
    st.rerun()


def _handle_label_action(repo_root: Path, state: dict[str, Any], params: dict[str, Any]) -> None:
    from mlops_mnist.ui.lifecycle import label_pending_batches

    labeled_now = label_pending_batches(state)
    if labeled_now <= 0:
        st.info("There is no new recorded data waiting for delayed labeling.")
        return

    with st.spinner("Applying delayed labels and updating supervised monitoring..."):
        monitoring_summary = run_lifecycle_monitoring(
            repo_root=repo_root,
            state=state,
            params_path=repo_root / "params.yaml",
        )

    message = (
        f"Labeled {labeled_now} samples. Rolling accuracy is now "
        f"{monitoring_summary['current_batch']['accuracy']:.4f}."
    )
    if monitoring_summary.get("alerts"):
        append_event_log(
            state,
            level="warning",
            message=message + " Supervised alerts: " + ", ".join(monitoring_summary["alerts"]),
        )
        _set_flash_message(
            "warning",
            message + " Alerts: " + ", ".join(monitoring_summary["alerts"]),
        )
    else:
        append_event_log(state, level="info", message=message + " No supervised alerts are active.")
        _set_flash_message("success", message)

    save_lifecycle_state(repo_root, state)
    st.rerun()


def _handle_clear_labeled_action(repo_root: Path, state: dict[str, Any]) -> None:
    cleared_samples = clear_labeled_batches(state)
    if cleared_samples <= 0:
        st.info("There are no labeled samples to clean from the lifecycle buffer.")
        return

    message = (
        f"Removed {cleared_samples} labeled samples from the lifecycle buffer. "
        "Supervised monitoring was cleared."
    )
    append_event_log(state, level="warning", message=message)
    _set_flash_message("warning", message)
    save_lifecycle_state(repo_root, state)
    st.rerun()


def _handle_retraining_action(
    repo_root: Path,
    api_base_url: str,
    state: dict[str, Any],
    params: dict[str, Any],
) -> None:
    min_recent_samples = int(params["retraining"]["min_recent_samples"])
    ready_labeled = count_ready_labeled_samples(state)
    if ready_labeled < min_recent_samples:
        st.info(
            f"Retraining needs at least {min_recent_samples} labeled samples. "
            f"Only {ready_labeled} are ready right now."
        )
        return

    with st.spinner("Training a challenger model from the labeled lifecycle buffer..."):
        with st.status("Training challenger model...", expanded=True) as status:
            status.write(f"Preparing {ready_labeled} labeled samples from the lifecycle buffer.")
            status.write("Launching retraining and challenger comparison.")
            retraining_summary = run_lifecycle_retraining(
                repo_root=repo_root,
                state=state,
                params_path=repo_root / "params.yaml",
                force_retrain=True,
            )
            decision = retraining_summary["decision"]
            status.write(
                "Finished challenger evaluation with adaptation gain "
                f"{decision['adaptation_accuracy_gain']:.4f} and benchmark regression "
                f"{decision['benchmark_accuracy_regression']:.4f}."
            )
            status.update(label="Retraining workflow finished.", state="complete")

    decision = retraining_summary["decision"]
    if decision["accepted"]:
        reload_summary = _reload_api_model(api_base_url)
        training_reference_examples = state.get("baseline_reference", {}).get("reference_data_examples")
        training_reference_message = (
            f" Training reference now tracks {training_reference_examples} examples."
            if training_reference_examples is not None
            else ""
        )
        append_event_log(
            state,
            level="success",
            message=(
                f"Challenger accepted with adaptation gain {decision['adaptation_accuracy_gain']:.4f} "
                f"and benchmark regression {decision['benchmark_accuracy_regression']:.4f}. "
                f"Production moved from version {retraining_summary['production_model']['version_before']} "
                f"to {retraining_summary['production_model']['version_after']}."
                f"{training_reference_message}"
            ),
        )
        save_lifecycle_state(repo_root, state)
        if reload_summary is None:
            _set_flash_message(
                "warning",
                "The challenger was promoted, but the API could not reload automatically. Restart the API to serve the new version."
                + training_reference_message,
            )
        else:
            _set_flash_message(
                "success",
                f"Challenger promoted to production version {reload_summary.get('model_version', retraining_summary['production_model']['version_after'])} and the API reloaded it."
                + training_reference_message,
            )
    else:
        append_event_log(
            state,
            level="warning",
            message=(
                f"Challenger rejected with adaptation gain {decision['adaptation_accuracy_gain']:.4f} "
                f"and benchmark regression {decision['benchmark_accuracy_regression']:.4f}. "
                f"Production stays on version {retraining_summary['production_model']['version_before']}."
            ),
        )
        save_lifecycle_state(repo_root, state)
        _set_flash_message(
            "warning",
            "The challenger did not beat the current production model, so production was preserved.",
        )

    st.rerun()


def _handle_initial_dvc_repro(repo_root: Path) -> None:
    with st.spinner("Running dvc repro to materialize the baseline data and scenarios..."):
        result = run_setup_command(
            repo_root,
            command=[*resolve_dvc_command(), "repro"],
            action_name="run_dvc_repro",
            success_message="dvc repro completed. Baseline data and scenario batches are ready.",
            failure_message="dvc repro failed.",
        )

    if result["ok"]:
        _load_batch.clear()
        _set_flash_message("success", "dvc repro completed. Baseline data and source batches are ready.")
    else:
        _set_flash_message("warning", "dvc repro failed. Open the latest setup command output on the Initial Setup page for details.")
    st.rerun()


def _handle_initial_training_and_promotion(repo_root: Path, api_base_url: str) -> None:
    with st.spinner("Training the initial CNN candidates..."):
        training_result = run_setup_command(
            repo_root,
            command=resolve_python_command("scripts/train_model.py"),
            action_name="run_initial_training",
            success_message="Initial competitor training finished.",
            failure_message="Initial competitor training failed.",
        )

    if not training_result["ok"]:
        _set_flash_message(
            "warning",
            "The initial training run failed. Open the latest setup command output on the Initial Setup page for details.",
        )
        st.rerun()

    with st.spinner("Promoting the winning initial model to production..."):
        promotion_result = run_setup_command(
            repo_root,
            command=resolve_python_command("scripts/promote_model.py"),
            action_name="promote_initial_winner",
            success_message="The winning initial model was promoted to production.",
            failure_message="The initial promotion step failed.",
            mode_on_success=SETUP_MODE_INITIALIZED,
        )

    if not promotion_result["ok"]:
        _set_flash_message(
            "warning",
            "The initial training run finished, but the promotion step failed. Open the latest setup command output on the Initial Setup page for details.",
        )
        st.rerun()

    training_summary = latest_training_summary(repo_root) or {}
    promotion_summary = latest_promotion_summary(repo_root) or {}
    best_candidate = training_summary.get("best_candidate") or promotion_summary.get("selected_result", {}).get("candidate_name")
    promoted_version = promotion_summary.get("registered_version", "unknown")
    reload_summary = _reload_api_model(api_base_url)
    if reload_summary is None:
        _set_flash_message(
            "warning",
            f"Initial training finished and promoted {best_candidate or 'the winning model'} to version {promoted_version}, but the API could not reload automatically.",
        )
    else:
        _set_flash_message(
            "success",
            f"Initial training finished. Promoted {best_candidate or 'the winning model'} to production version {reload_summary.get('model_version', promoted_version)} and reloaded the API.",
        )
    st.rerun()


def _handle_pre_setup_reset(repo_root: Path, api_base_url: str) -> None:
    with st.spinner("Removing generated data, models, and MLflow state..."):
        reset_to_pre_setup_state(repo_root=repo_root, dvc_path=repo_root / "dvc.yaml")

    _load_batch.clear()
    reload_summary = _reload_api_model(api_base_url)
    if reload_summary is None:
        _set_flash_message(
            "success",
            "Hard reset complete. The repository is back to the empty pre-setup state.",
        )
    else:
        _set_flash_message(
            "success",
            "Hard reset complete. The repository is back to the empty pre-setup state and the running API was refreshed.",
        )
    st.rerun()


def _queue_full_system_reset() -> None:
    st.session_state[PENDING_FULL_RESET_KEY] = True


def _consume_pending_full_system_reset(repo_root: Path, api_base_url: str) -> None:
    if not st.session_state.pop(PENDING_FULL_RESET_KEY, False):
        return

    with st.spinner("Restoring the app to its initial state..."):
        reset_summary = reset_application_state(
            repo_root=repo_root,
            params_path=repo_root / "params.yaml",
        )

    restored_batch_name = reset_summary["lifecycle"]["active_batch_name"]
    restored_batch_size = int(reset_summary["lifecycle"]["batch_size"])
    st.session_state["lifecycle_batch_name"] = restored_batch_name
    st.session_state["lifecycle_batch_size"] = restored_batch_size

    reload_summary = _reload_api_model(api_base_url)
    if reload_summary is None:
        _set_flash_message(
            "warning",
            "The app was reset to the initial production-v1 state, but the API could not reload automatically. Restart or reload the API to serve version 1.",
        )
    else:
        _set_flash_message(
            "success",
            f"Full reset complete. The app is back on production version {reload_summary.get('model_version', '1')} with the default lifecycle source {restored_batch_name}.",
        )


def _build_preview_payload(
    slice_payload: dict[str, Any],
    labels: np.ndarray,
    prediction: dict[str, Any],
    latency_ms: float,
    labels_hidden: bool,
) -> dict[str, Any]:
    predicted_labels = [int(row["predicted_label"]) for row in prediction["predictions"]]
    confidences = [float(row["confidence"]) for row in prediction["predictions"]]
    accuracy = float(np.mean(np.asarray(predicted_labels) == labels)) if len(labels) else 0.0

    results = []
    for offset, predicted_label in enumerate(predicted_labels):
        row = {
            "index": int(slice_payload["indices"][offset]),
            "predicted_label": int(predicted_label),
            "confidence": float(confidences[offset]),
        }
        if not labels_hidden:
            row["true_label"] = int(labels[offset])
        results.append(row)

    return {
        "step": int(slice_payload["step"]),
        "start_index": int(slice_payload["start_index"]),
        "end_index": int(slice_payload["end_index"]),
        "batch_size": len(slice_payload["indices"]),
        "labels_hidden": bool(labels_hidden),
        "mean_confidence": float(prediction["mean_confidence"]),
        "accuracy": float(accuracy),
        "latency_ms": float(latency_ms),
        "model_version": prediction["model_version"],
        "results": results,
    }


def _render_latest_preview(preview_payload: dict[str, Any] | None) -> None:
    st.markdown("#### Latest Batch Result")
    if preview_payload is None:
        st.caption("Click Run Next Batch to simulate the next incoming slice of data.")
        return

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Simulated step", str(preview_payload["step"]))
    metric_col2.metric(
        "Batch range",
        f"{preview_payload['start_index']}-{preview_payload['end_index'] - 1}",
    )
    metric_col3.metric("Mean confidence", f"{preview_payload['mean_confidence']:.4f}")
    metric_col4.metric("Latency (ms)", f"{preview_payload['latency_ms']:.1f}")

    if preview_payload["labels_hidden"]:
        st.info("This batch is being treated as unlabeled live data. Accuracy and F1 stay hidden until Label Data is clicked.")
    else:
        st.success(f"Preview batch accuracy: {preview_payload['accuracy']:.4f}")

    st.dataframe(preview_payload["results"], use_container_width=True)


def _render_alert_panel(state: dict[str, Any], params: dict[str, Any]) -> None:
    st.markdown("#### Action Thresholds")
    threshold_cols = st.columns(3)
    thresholds = params["monitoring"]["thresholds"]
    threshold_cols[0].metric("Min rolling accuracy", f"{float(thresholds['min_rolling_accuracy']):.2f}")
    threshold_cols[1].metric("Max confidence drop", f"{float(thresholds['max_confidence_drop']):.2f}")
    threshold_cols[2].metric(
        "Max histogram JS",
        f"{float(thresholds['max_histogram_js_divergence']):.2f}",
    )

    current_alerts = state.get("current_alerts", [])
    if current_alerts:
        for alert in current_alerts:
            st.warning(f"Alert trigger: {alert}")
    else:
        st.caption("No active threshold violations in the current lifecycle state.")


def _render_placeholder_dashboards(state: dict[str, Any]) -> None:
    st.markdown("#### Monitoring Placeholders")
    st.caption("Simulated time is click-driven here. These charts stand in for Prometheus and Grafana until those services are available.")

    prom_col, graf_col = st.columns(2)
    with prom_col:
        st.markdown("##### Prometheus Placeholder")
        if state.get("unsupervised_history"):
            unsupervised = pd.DataFrame(state["unsupervised_history"]).set_index("step")
            if len(unsupervised) > 1:
                st.line_chart(unsupervised[["recorded_examples"]], height=180)
                st.line_chart(unsupervised[["mean_confidence", "confidence_drop"]], height=180)
                st.line_chart(unsupervised[["histogram_js_divergence"]], height=180)
                st.line_chart(unsupervised[["latency_ms"]], height=180)
            else:
                summary_row = unsupervised.iloc[-1]
                metric_cols = st.columns(4)
                metric_cols[0].metric("Recorded examples", int(summary_row["recorded_examples"]))
                metric_cols[1].metric("Mean confidence", f"{float(summary_row['mean_confidence']):.4f}")
                metric_cols[2].metric("Confidence drop", f"{float(summary_row['confidence_drop']):.4f}")
                metric_cols[3].metric("Latency (ms)", f"{float(summary_row['latency_ms']):.1f}")
            distribution = pd.DataFrame(
                {"latest_recorded_batch": latest_recorded_distribution(state)},
                index=[str(index) for index in range(10)],
            )
            st.bar_chart(distribution)
        else:
            st.info("Turn Record on and run batches to populate the unsupervised lifecycle charts.")

    with graf_col:
        st.markdown("##### Grafana Placeholder")
        if state.get("supervised_history"):
            supervised = pd.DataFrame(state["supervised_history"]).set_index("step")
            if len(supervised) > 1:
                st.line_chart(supervised[["accuracy", "macro_f1"]], height=180)
                st.line_chart(supervised[["accuracy_drop", "confidence_drop"]], height=180)
                st.line_chart(supervised[["histogram_js_divergence"]], height=180)
            else:
                summary_row = supervised.iloc[-1]
                metric_cols = st.columns(4)
                metric_cols[0].metric("Rolling accuracy", f"{float(summary_row['accuracy']):.4f}")
                metric_cols[1].metric("Macro F1", f"{float(summary_row['macro_f1']):.4f}")
                metric_cols[2].metric("Accuracy drop", f"{float(summary_row['accuracy_drop']):.4f}")
                metric_cols[3].metric(
                    "Histogram JS",
                    f"{float(summary_row['histogram_js_divergence']):.4f}",
                )
        else:
            st.info("Click Label Data after recording batches to reveal supervised lifecycle metrics.")

        if state.get("retraining_history"):
            st.markdown("##### Retraining Decisions")
            retraining_history = pd.DataFrame(list(reversed(state["retraining_history"][-5:])))
            retraining_history = retraining_history.drop(columns=["accuracy_gain"], errors="ignore")
            retraining_columns = [
                "step",
                "accepted",
                "action",
                "reason",
                "adaptation_accuracy_gain",
                "benchmark_accuracy_regression",
                "production_adaptation_accuracy",
                "challenger_adaptation_accuracy",
                "production_benchmark_accuracy",
                "challenger_benchmark_accuracy",
                "adaptation_test_examples",
                "benchmark_test_examples",
                "version_before",
                "version_after",
                "summary_path",
            ]
            retraining_history = retraining_history[
                [column for column in retraining_columns if column in retraining_history.columns]
            ]
            st.dataframe(retraining_history, use_container_width=True)
        if state.get("training_reference_history"):
            st.markdown("##### Training Reference Versions")
            training_reference_history = pd.DataFrame(list(reversed(state["training_reference_history"][-5:])))
            training_reference_columns = [
                "step",
                "model_version",
                "training_reference_name",
                "training_reference_examples_before",
                "training_reference_examples",
                "recent_train_examples",
                "recent_adaptation_test_examples",
                "recent_examples_absorbed",
                "source_batch_name",
                "dvc_tracked",
                "dvc_file_path",
                "summary_path",
            ]
            training_reference_history = training_reference_history[
                [column for column in training_reference_columns if column in training_reference_history.columns]
            ]
            st.dataframe(training_reference_history, use_container_width=True)


def _render_event_log(state: dict[str, Any]) -> None:
    st.markdown("#### Event Timeline")
    if not state.get("event_log"):
        st.caption("Lifecycle messages will appear here as you preview, record, label, and retrain.")
        return

    timeline = list(reversed(state["event_log"][-20:]))
    st.dataframe(timeline, use_container_width=True)


def _render_flash_message() -> None:
    flash_message = st.session_state.pop(FLASH_MESSAGE_KEY, None)
    if flash_message is None:
        return
    level = flash_message.get("level", "info")
    message = flash_message.get("message", "")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _set_flash_message(level: str, message: str) -> None:
    st.session_state[FLASH_MESSAGE_KEY] = {"level": level, "message": message}


def _load_images_from_path(repo_root: Path, batch_path: str | Path) -> np.ndarray:
    resolved_path = resolve_repo_bound_path(repo_root, batch_path)
    with np.load(resolved_path) as batch:
        return batch["images"].astype(np.float32)


def _prepare_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


if __name__ == "__main__":
    main()