"""
test_convert_tflite.py - trains a small model, converts it, and validates
the quantized model is both small enough for a microcontroller and still
detects anomalies correctly.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import tensorflow as tf  # noqa: E402
from train_autoencoder import build_autoencoder, normalize  # noqa: E402
from convert_tflite import convert, run_quantized_inference  # noqa: E402
from vibration_signals import generate_dataset  # noqa: E402

MAX_MODEL_SIZE_BYTES = 50_000  # comfortably fits ESP32 flash/RAM budgets


@pytest.fixture(scope="module")
def quantized_model_and_data(tmp_path_factory):
    tf.random.set_seed(1)
    normal, anomalies, _ = generate_dataset(n_normal=300, n_anomaly_per_type=30, seed=1)
    mean, std = float(normal.mean()), float(normal.std())
    normal_norm = normalize(normal, mean, std)
    anomalies_norm = normalize(anomalies, mean, std)

    model = build_autoencoder()
    model.fit(normal_norm, normal_norm, epochs=15, batch_size=32, verbose=0)

    tmp_dir = tmp_path_factory.mktemp("model")
    model_path = str(tmp_dir / "model.keras")
    model.save(model_path)

    tflite_bytes = convert(model_path, normal_norm)
    return tflite_bytes, normal_norm, anomalies_norm


def test_quantized_model_is_small_enough_for_a_microcontroller(quantized_model_and_data):
    tflite_bytes, _, _ = quantized_model_and_data
    assert len(tflite_bytes) < MAX_MODEL_SIZE_BYTES


def test_quantized_model_has_valid_tflite_magic_bytes(quantized_model_and_data):
    tflite_bytes, _, _ = quantized_model_and_data
    # every valid .tflite (flatbuffer) file has "TFL3" at offset 4
    assert tflite_bytes[4:8] == b"TFL3"


def test_quantized_inference_preserves_anomaly_detection(quantized_model_and_data):
    tflite_bytes, normal_norm, anomalies_norm = quantized_model_and_data
    normal_err = run_quantized_inference(tflite_bytes, normal_norm)
    anomaly_err = run_quantized_inference(tflite_bytes, anomalies_norm)

    assert np.all(np.isfinite(normal_err))
    assert np.all(np.isfinite(anomaly_err))

    threshold = float(np.percentile(normal_err, 99.0))
    false_positive_rate = float(np.mean(normal_err > threshold))
    recall = float(np.mean(anomaly_err > threshold))

    assert false_positive_rate <= 0.05
    assert recall >= 0.8, "quantization should not meaningfully degrade anomaly detection"
