from __future__ import annotations

from threading import Lock
from typing import Any, Mapping

from prometheus_client import Gauge, start_http_server


DEFAULT_LIFECYCLE_METRICS_PORT = 9108

UNSUPERVISED_RECORDED_EXAMPLES = Gauge(
    "mnist_lifecycle_recorded_examples",
    "Number of recorded lifecycle examples currently buffered.",
)
UNSUPERVISED_MEAN_CONFIDENCE = Gauge(
    "mnist_lifecycle_unsupervised_mean_confidence",
    "Mean confidence observed in the latest recorded lifecycle batch.",
)
UNSUPERVISED_LATENCY_MS = Gauge(
    "mnist_lifecycle_unsupervised_latency_ms",
    "Latency in milliseconds observed in the latest recorded lifecycle batch.",
)
UNSUPERVISED_HISTOGRAM_JS_DIVERGENCE = Gauge(
    "mnist_lifecycle_unsupervised_histogram_js_divergence",
    "Pixel histogram Jensen-Shannon divergence for the latest recorded lifecycle batch.",
)
UNSUPERVISED_CONFIDENCE_DROP = Gauge(
    "mnist_lifecycle_unsupervised_confidence_drop",
    "Confidence drop for the latest recorded lifecycle batch versus the baseline reference.",
)
LATEST_RECORDED_CLASS_EXAMPLES = Gauge(
    "mnist_lifecycle_latest_recorded_class_examples",
    "Predicted class counts for the latest recorded lifecycle batch.",
    ["digit"],
)

SUPERVISED_LABELED_EXAMPLES = Gauge(
    "mnist_lifecycle_labeled_examples",
    "Number of labeled lifecycle examples currently available for evaluation.",
)
SUPERVISED_ACCURACY = Gauge(
    "mnist_lifecycle_supervised_accuracy",
    "Accuracy measured on the latest labeled lifecycle batch.",
)
SUPERVISED_MACRO_F1 = Gauge(
    "mnist_lifecycle_supervised_macro_f1",
    "Macro F1 measured on the latest labeled lifecycle batch.",
)
SUPERVISED_MEAN_CONFIDENCE = Gauge(
    "mnist_lifecycle_supervised_mean_confidence",
    "Mean confidence measured on the latest labeled lifecycle batch.",
)
SUPERVISED_ACCURACY_DROP = Gauge(
    "mnist_lifecycle_supervised_accuracy_drop",
    "Accuracy drop measured on the latest labeled lifecycle batch versus the baseline reference.",
)
SUPERVISED_CONFIDENCE_DROP = Gauge(
    "mnist_lifecycle_supervised_confidence_drop",
    "Confidence drop measured on the latest labeled lifecycle batch versus the baseline reference.",
)
SUPERVISED_HISTOGRAM_JS_DIVERGENCE = Gauge(
    "mnist_lifecycle_supervised_histogram_js_divergence",
    "Pixel histogram Jensen-Shannon divergence for the latest labeled lifecycle batch.",
)

RETRAINING_ACCEPTED = Gauge(
    "mnist_lifecycle_retraining_accepted",
    "Latest lifecycle retraining decision encoded as 1 for accepted and 0 for rejected.",
)
RETRAINING_ADAPTATION_ACCURACY_GAIN = Gauge(
    "mnist_lifecycle_retraining_adaptation_accuracy_gain",
    "Adaptation accuracy delta between challenger and production on the latest lifecycle retraining run.",
)
RETRAINING_BENCHMARK_ACCURACY_REGRESSION = Gauge(
    "mnist_lifecycle_retraining_benchmark_accuracy_regression",
    "Signed benchmark accuracy regression measured on the latest lifecycle retraining run.",
)

TRAINING_REFERENCE_EXAMPLES = Gauge(
    "mnist_lifecycle_training_reference_examples",
    "Number of examples in the current promoted lifecycle training reference.",
)
TRAINING_REFERENCE_EXAMPLES_BEFORE = Gauge(
    "mnist_lifecycle_training_reference_examples_before",
    "Number of examples in the previous lifecycle training reference before the latest promotion.",
)
TRAINING_REFERENCE_RECENT_TRAIN_EXAMPLES = Gauge(
    "mnist_lifecycle_recent_train_examples",
    "Number of recent lifecycle examples used for fitting before the latest promotion.",
)
TRAINING_REFERENCE_RECENT_ADAPTATION_TEST_EXAMPLES = Gauge(
    "mnist_lifecycle_recent_adaptation_test_examples",
    "Number of recent lifecycle examples reserved for adaptation testing before the latest promotion.",
)
TRAINING_REFERENCE_RECENT_EXAMPLES_ABSORBED = Gauge(
    "mnist_lifecycle_recent_examples_absorbed",
    "Number of recent lifecycle examples absorbed into the latest promoted training reference.",
)

_SCALAR_GAUGES = (
    UNSUPERVISED_RECORDED_EXAMPLES,
    UNSUPERVISED_MEAN_CONFIDENCE,
    UNSUPERVISED_LATENCY_MS,
    UNSUPERVISED_HISTOGRAM_JS_DIVERGENCE,
    UNSUPERVISED_CONFIDENCE_DROP,
    SUPERVISED_LABELED_EXAMPLES,
    SUPERVISED_ACCURACY,
    SUPERVISED_MACRO_F1,
    SUPERVISED_MEAN_CONFIDENCE,
    SUPERVISED_ACCURACY_DROP,
    SUPERVISED_CONFIDENCE_DROP,
    SUPERVISED_HISTOGRAM_JS_DIVERGENCE,
    RETRAINING_ACCEPTED,
    RETRAINING_ADAPTATION_ACCURACY_GAIN,
    RETRAINING_BENCHMARK_ACCURACY_REGRESSION,
    TRAINING_REFERENCE_EXAMPLES,
    TRAINING_REFERENCE_EXAMPLES_BEFORE,
    TRAINING_REFERENCE_RECENT_TRAIN_EXAMPLES,
    TRAINING_REFERENCE_RECENT_ADAPTATION_TEST_EXAMPLES,
    TRAINING_REFERENCE_RECENT_EXAMPLES_ABSORBED,
)

_SERVER_LOCK = Lock()
_STARTED_PORTS: set[int] = set()


def ensure_lifecycle_metrics_server(port: int = DEFAULT_LIFECYCLE_METRICS_PORT) -> bool:
    # Start the lifecycle metrics server only once per port.
    normalized_port = int(port)
    with _SERVER_LOCK:
        if normalized_port in _STARTED_PORTS:
            return True
        try:
            start_http_server(normalized_port)
        except OSError:
            return False
        _STARTED_PORTS.add(normalized_port)
        return True


def reset_lifecycle_metrics() -> None:
    # Clear all exported lifecycle gauges before rewriting them.
    for gauge in _SCALAR_GAUGES:
        gauge.set(0.0)
    for digit in range(10):
        LATEST_RECORDED_CLASS_EXAMPLES.labels(digit=str(digit)).set(0.0)


def sync_lifecycle_state_metrics(state: Mapping[str, Any] | None) -> None:
    # Map the saved lifecycle state into Prometheus gauge values.
    reset_lifecycle_metrics()
    if not state:
        return

    # Publish the latest unsupervised monitoring values.
    unsupervised_history = list(state.get("unsupervised_history") or [])
    if unsupervised_history:
        latest_unsupervised = unsupervised_history[-1]
        UNSUPERVISED_RECORDED_EXAMPLES.set(float(latest_unsupervised.get("recorded_examples", 0)))
        UNSUPERVISED_MEAN_CONFIDENCE.set(float(latest_unsupervised.get("mean_confidence", 0.0)))
        UNSUPERVISED_LATENCY_MS.set(float(latest_unsupervised.get("latency_ms", 0.0)))
        UNSUPERVISED_HISTOGRAM_JS_DIVERGENCE.set(float(latest_unsupervised.get("histogram_js_divergence", 0.0)))
        UNSUPERVISED_CONFIDENCE_DROP.set(float(latest_unsupervised.get("confidence_drop", 0.0)))

    recorded_batches = list(state.get("recorded_batches") or [])
    if recorded_batches:
        # Publish the latest predicted class counts.
        predicted_labels = list(recorded_batches[-1].get("predicted_labels") or [])
        digit_counts = [0] * 10
        for raw_label in predicted_labels:
            try:
                digit = int(raw_label)
            except (TypeError, ValueError):
                continue
            if 0 <= digit <= 9:
                digit_counts[digit] += 1
        for digit, count in enumerate(digit_counts):
            LATEST_RECORDED_CLASS_EXAMPLES.labels(digit=str(digit)).set(float(count))

    supervised_history = list(state.get("supervised_history") or [])
    if supervised_history:
        # Publish the latest supervised monitoring values.
        latest_supervised = supervised_history[-1]
        SUPERVISED_LABELED_EXAMPLES.set(float(latest_supervised.get("labeled_examples", 0)))
        SUPERVISED_ACCURACY.set(float(latest_supervised.get("accuracy", 0.0)))
        SUPERVISED_MACRO_F1.set(float(latest_supervised.get("macro_f1", 0.0)))
        SUPERVISED_MEAN_CONFIDENCE.set(float(latest_supervised.get("mean_confidence", 0.0)))
        SUPERVISED_ACCURACY_DROP.set(float(latest_supervised.get("accuracy_drop", 0.0)))
        SUPERVISED_CONFIDENCE_DROP.set(float(latest_supervised.get("confidence_drop", 0.0)))
        SUPERVISED_HISTOGRAM_JS_DIVERGENCE.set(
            float(latest_supervised.get("histogram_js_divergence", 0.0))
        )

    retraining_history = list(state.get("retraining_history") or [])
    if retraining_history:
        # Publish the most recent retraining decision.
        latest_retraining = retraining_history[-1]
        RETRAINING_ACCEPTED.set(1.0 if latest_retraining.get("accepted") else 0.0)
        RETRAINING_ADAPTATION_ACCURACY_GAIN.set(
            float(latest_retraining.get("adaptation_accuracy_gain", 0.0))
        )
        RETRAINING_BENCHMARK_ACCURACY_REGRESSION.set(
            float(latest_retraining.get("benchmark_accuracy_regression", 0.0))
        )

    training_reference_history = list(state.get("training_reference_history") or [])
    if training_reference_history:
        # Publish the size changes in the promoted training reference.
        latest_training_reference = training_reference_history[-1]
        TRAINING_REFERENCE_EXAMPLES.set(
            float(latest_training_reference.get("training_reference_examples", 0))
        )
        TRAINING_REFERENCE_EXAMPLES_BEFORE.set(
            float(latest_training_reference.get("training_reference_examples_before", 0))
        )
        TRAINING_REFERENCE_RECENT_TRAIN_EXAMPLES.set(
            float(latest_training_reference.get("recent_train_examples", 0))
        )
        TRAINING_REFERENCE_RECENT_ADAPTATION_TEST_EXAMPLES.set(
            float(latest_training_reference.get("recent_adaptation_test_examples", 0))
        )
        TRAINING_REFERENCE_RECENT_EXAMPLES_ABSORBED.set(
            float(latest_training_reference.get("recent_examples_absorbed", 0))
        )


def collect_lifecycle_metrics_snapshot() -> dict[str, Any]:
    # Return the current gauge values for tests and manual checks.
    return {
        "recorded_examples": UNSUPERVISED_RECORDED_EXAMPLES._value.get(),
        "unsupervised_mean_confidence": UNSUPERVISED_MEAN_CONFIDENCE._value.get(),
        "unsupervised_latency_ms": UNSUPERVISED_LATENCY_MS._value.get(),
        "unsupervised_histogram_js_divergence": UNSUPERVISED_HISTOGRAM_JS_DIVERGENCE._value.get(),
        "unsupervised_confidence_drop": UNSUPERVISED_CONFIDENCE_DROP._value.get(),
        "latest_recorded_class_examples": {
            str(digit): LATEST_RECORDED_CLASS_EXAMPLES.labels(digit=str(digit))._value.get()
            for digit in range(10)
        },
        "labeled_examples": SUPERVISED_LABELED_EXAMPLES._value.get(),
        "supervised_accuracy": SUPERVISED_ACCURACY._value.get(),
        "supervised_macro_f1": SUPERVISED_MACRO_F1._value.get(),
        "supervised_mean_confidence": SUPERVISED_MEAN_CONFIDENCE._value.get(),
        "supervised_accuracy_drop": SUPERVISED_ACCURACY_DROP._value.get(),
        "supervised_confidence_drop": SUPERVISED_CONFIDENCE_DROP._value.get(),
        "supervised_histogram_js_divergence": SUPERVISED_HISTOGRAM_JS_DIVERGENCE._value.get(),
        "retraining_accepted": RETRAINING_ACCEPTED._value.get(),
        "retraining_adaptation_accuracy_gain": RETRAINING_ADAPTATION_ACCURACY_GAIN._value.get(),
        "retraining_benchmark_accuracy_regression": RETRAINING_BENCHMARK_ACCURACY_REGRESSION._value.get(),
        "training_reference_examples": TRAINING_REFERENCE_EXAMPLES._value.get(),
        "training_reference_examples_before": TRAINING_REFERENCE_EXAMPLES_BEFORE._value.get(),
        "recent_train_examples": TRAINING_REFERENCE_RECENT_TRAIN_EXAMPLES._value.get(),
        "recent_adaptation_test_examples": TRAINING_REFERENCE_RECENT_ADAPTATION_TEST_EXAMPLES._value.get(),
        "recent_examples_absorbed": TRAINING_REFERENCE_RECENT_EXAMPLES_ABSORBED._value.get(),
    }