"""
vibration_signals.py - synthetic accelerometer-style vibration windows,
shared by the training dataset generator (data/generate_dataset.py) and
the live simulator (simulator/virtual_sensor.py) so both draw from the
exact same signal model.

Design choice, stated plainly: this simulates the *shape* of common
rotating-machinery fault signatures (bearing impacts, imbalance,
misalignment, looseness) at a scale convenient for a fast demo (200Hz,
64-sample windows) - it is not a physically calibrated vibration model.
Real bearing fault frequencies are typically captured at kHz sample rates
over much longer windows; this project demonstrates the ML/TinyML
*pipeline* (train -> quantize -> deploy -> detect), not a certified
condition-monitoring instrument.

Each window is WINDOW_SIZE samples representing SAMPLE_RATE_HZ Hz of a
single accelerometer axis, roughly normalized to [-1, 1].
"""
import numpy as np

WINDOW_SIZE = 64
SAMPLE_RATE_HZ = 200.0
BASE_FREQ_HZ = 10.0  # simulated shaft rotation frequency

FAULT_TYPES = ["bearing", "imbalance", "misalignment", "looseness"]

_T = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ


def generate_normal_window(rng: np.random.Generator) -> np.ndarray:
    """Steady rotation: fundamental + a couple of weak harmonics + noise."""
    amp_jitter = rng.uniform(0.9, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    signal = amp_jitter * np.sin(2 * np.pi * BASE_FREQ_HZ * _T + phase)
    signal += 0.08 * np.sin(2 * np.pi * 2 * BASE_FREQ_HZ * _T + phase)
    signal += 0.04 * np.sin(2 * np.pi * 3 * BASE_FREQ_HZ * _T + phase)
    signal += rng.normal(0, 0.05, WINDOW_SIZE)
    return signal.astype(np.float32)


def generate_anomaly_window(rng: np.random.Generator, fault_type: str) -> np.ndarray:
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unknown fault_type {fault_type!r}, expected one of {FAULT_TYPES}")

    signal = generate_normal_window(rng)

    if fault_type == "bearing":
        # sharp periodic impacts, faster than the shaft rotation
        impact_period = max(4, WINDOW_SIZE // 8)
        for start in range(0, WINDOW_SIZE, impact_period):
            width = 2
            end = min(WINDOW_SIZE, start + width)
            signal[start:end] += rng.uniform(1.2, 2.0)

    elif fault_type == "imbalance":
        # dominant amplitude increase at the fundamental frequency
        phase = rng.uniform(0, 2 * np.pi)
        signal += rng.uniform(0.8, 1.4) * np.sin(2 * np.pi * BASE_FREQ_HZ * _T + phase)

    elif fault_type == "misalignment":
        # strong 2x-harmonic component (classic misalignment signature)
        phase = rng.uniform(0, 2 * np.pi)
        signal += rng.uniform(0.7, 1.2) * np.sin(2 * np.pi * 2 * BASE_FREQ_HZ * _T + phase)

    elif fault_type == "looseness":
        # broadband high-frequency noise increase
        signal += rng.normal(0, rng.uniform(0.3, 0.5), WINDOW_SIZE)

    return signal.astype(np.float32)


def generate_dataset(n_normal: int, n_anomaly_per_type: int, seed: int = 42):
    """Returns (normal_windows, anomaly_windows, anomaly_labels) as numpy
    arrays - anomaly_labels holds the fault_type string per anomaly row."""
    rng = np.random.default_rng(seed)

    normal = np.stack([generate_normal_window(rng) for _ in range(n_normal)])

    anomalies = []
    labels = []
    for fault_type in FAULT_TYPES:
        for _ in range(n_anomaly_per_type):
            anomalies.append(generate_anomaly_window(rng, fault_type))
            labels.append(fault_type)
    anomalies = np.stack(anomalies) if anomalies else np.empty((0, WINDOW_SIZE), dtype=np.float32)

    return normal, anomalies, np.array(labels)
