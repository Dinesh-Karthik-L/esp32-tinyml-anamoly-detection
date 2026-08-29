"""
test_train_autoencoder.py - trains a small, fast model (few epochs, small
dataset) to validate the training PIPELINE and evaluation logic, not to
reproduce the fully-tuned artifacts in model/artifacts/ (those were
produced by the slower, full run documented in the README).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import tensorflow as tf  # noqa: E402
from train_autoencoder import build_autoencoder, normalize, reconstruction_error, evaluate  # noqa: E402
from vibration_signals import generate_dataset, WINDOW_SIZE  # noqa: E402


@pytest.fixture(scope="module")
def trained_small_model():
    tf.random.set_seed(0)
    normal, anomalies, _ = generate_dataset(n_normal=300, n_anomaly_per_type=30, seed=0)

    mean, std = float(normal.mean()), float(normal.std())
    normal_norm = normalize(normal, mean, std)
    anomalies_norm = normalize(anomalies, mean, std)

    model = build_autoencoder()
    model.fit(normal_norm, normal_norm, epochs=15, batch_size=32, verbose=0)
    return model, normal_norm, anomalies_norm


def test_build_autoencoder_has_expected_io_shape():
    model = build_autoencoder()
    assert model.input_shape == (None, WINDOW_SIZE)
    assert model.output_shape == (None, WINDOW_SIZE)


def test_normalize_produces_zero_mean_unit_std_on_training_data():
    data = np.random.default_rng(0).normal(5, 2, (100, WINDOW_SIZE)).astype(np.float32)
    mean, std = float(data.mean()), float(data.std())
    normed = normalize(data, mean, std)
    assert abs(normed.mean()) < 1e-5
    assert abs(normed.std() - 1.0) < 1e-5


def test_training_loss_is_finite_and_decreases(trained_small_model):
    model, normal_norm, _ = trained_small_model
    # loss should be small and finite - training on its own reconstruction
    # target is about as easy as it gets, so this mostly guards against NaNs
    err = reconstruction_error(model, normal_norm)
    assert np.all(np.isfinite(err))
    assert err.mean() < 1.0


def test_autoencoder_separates_anomalies_from_normal(trained_small_model):
    model, normal_norm, anomalies_norm = trained_small_model
    normal_err = reconstruction_error(model, normal_norm)
    anomaly_err = reconstruction_error(model, anomalies_norm)
    # the core claim of the whole project: anomalies reconstruct worse
    assert anomaly_err.mean() > normal_err.mean() * 3


def test_evaluate_reports_low_false_positive_rate_at_99th_percentile(trained_small_model):
    model, normal_norm, anomalies_norm = trained_small_model
    metrics = evaluate(model, normal_norm, anomalies_norm, threshold_percentile=99.0)
    # by construction, thresholding at the 99th percentile of normal error
    # should flag roughly 1% of normal windows as false positives
    assert metrics["false_positive_rate"] <= 0.05
    assert metrics["recall"] >= 0.8  # even with a small/fast model, most anomalies should stand out
