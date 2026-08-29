import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

from vibration_signals import (  # noqa: E402
    generate_normal_window, generate_anomaly_window, generate_dataset,
    FAULT_TYPES, WINDOW_SIZE,
)


def test_normal_window_has_expected_shape_and_dtype():
    rng = np.random.default_rng(0)
    w = generate_normal_window(rng)
    assert w.shape == (WINDOW_SIZE,)
    assert w.dtype == np.float32


def test_normal_window_is_reproducible_with_same_seed():
    w1 = generate_normal_window(np.random.default_rng(42))
    w2 = generate_normal_window(np.random.default_rng(42))
    np.testing.assert_array_equal(w1, w2)


def test_anomaly_window_rejects_unknown_fault_type():
    with pytest.raises(ValueError):
        generate_anomaly_window(np.random.default_rng(0), "warp_core_breach")


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_anomaly_windows_are_statistically_distinct_from_normal(fault_type):
    rng = np.random.default_rng(7)
    normal = np.stack([generate_normal_window(rng) for _ in range(300)])
    anomalies = np.stack([generate_anomaly_window(rng, fault_type) for _ in range(300)])

    normal_peak = np.abs(normal).max(axis=1).mean()
    anomaly_peak = np.abs(anomalies).max(axis=1).mean()
    # every fault signature is designed to increase peak amplitude or
    # broadband energy noticeably over normal operation
    assert anomaly_peak > normal_peak * 1.15, (
        f"{fault_type}: anomaly peak {anomaly_peak:.3f} not clearly above normal {normal_peak:.3f}"
    )


def test_generate_dataset_shapes_and_labels():
    normal, anomalies, labels = generate_dataset(n_normal=50, n_anomaly_per_type=10, seed=3)
    assert normal.shape == (50, WINDOW_SIZE)
    assert anomalies.shape == (40, WINDOW_SIZE)  # 4 fault types x 10
    assert len(labels) == 40
    assert set(labels) == set(FAULT_TYPES)
    for ft in FAULT_TYPES:
        assert (labels == ft).sum() == 10


def test_generate_dataset_is_reproducible_with_seed():
    n1, a1, l1 = generate_dataset(n_normal=20, n_anomaly_per_type=5, seed=99)
    n2, a2, l2 = generate_dataset(n_normal=20, n_anomaly_per_type=5, seed=99)
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(l1, l2)
