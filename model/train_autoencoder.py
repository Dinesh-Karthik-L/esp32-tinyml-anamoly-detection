"""
train_autoencoder.py - trains a small dense autoencoder on NORMAL vibration
windows only (unsupervised anomaly detection: we never show the model a
single anomaly during training - it just has to learn to reconstruct
"normal" well, and anomalies naturally reconstruct poorly).

Saves to model/artifacts/:
    autoencoder.keras       - the trained model
    normalization.json      - {mean, std} used to normalize windows
    threshold.json           - the reconstruction-error threshold + eval metrics

Usage:
    python model/train_autoencoder.py
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from vibration_signals import WINDOW_SIZE  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def build_autoencoder(window_size: int = WINDOW_SIZE) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size,), name="vibration_window")
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(8, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(window_size, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs, name="vibration_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def normalize(windows: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (windows - mean) / std


def reconstruction_error(model: tf.keras.Model, windows_norm: np.ndarray) -> np.ndarray:
    recon = model.predict(windows_norm, verbose=0)
    return np.mean((windows_norm - recon) ** 2, axis=1)


def evaluate(model, normal_val_norm, anomalies_norm, threshold_percentile: float = 99.0):
    normal_err = reconstruction_error(model, normal_val_norm)
    anomaly_err = reconstruction_error(model, anomalies_norm) if len(anomalies_norm) else np.array([])

    threshold = float(np.percentile(normal_err, threshold_percentile))

    false_positive_rate = float(np.mean(normal_err > threshold))
    recall = float(np.mean(anomaly_err > threshold)) if len(anomaly_err) else None

    return {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "normal_error_mean": float(normal_err.mean()),
        "normal_error_p99": float(np.percentile(normal_err, 99)),
        "anomaly_error_mean": float(anomaly_err.mean()) if len(anomaly_err) else None,
        "false_positive_rate": false_positive_rate,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the vibration anomaly-detection autoencoder")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    normal, anomalies = data["normal"], data["anomalies"]

    n_val = int(len(normal) * args.val_split)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(normal))
    normal = normal[perm]
    normal_train, normal_val = normal[n_val:], normal[:n_val]

    mean, std = float(normal_train.mean()), float(normal_train.std())
    normal_train_norm = normalize(normal_train, mean, std)
    normal_val_norm = normalize(normal_val, mean, std)
    anomalies_norm = normalize(anomalies, mean, std)

    model = build_autoencoder()
    history = model.fit(normal_train_norm, normal_train_norm,
                         validation_data=(normal_val_norm, normal_val_norm),
                         epochs=args.epochs, batch_size=32, verbose=0)

    metrics = evaluate(model, normal_val_norm, anomalies_norm)
    metrics["final_train_loss"] = float(history.history["loss"][-1])
    metrics["final_val_loss"] = float(history.history["val_loss"][-1])

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save(os.path.join(ARTIFACTS_DIR, "autoencoder.keras"))
    with open(os.path.join(ARTIFACTS_DIR, "normalization.json"), "w") as f:
        json.dump({"mean": mean, "std": std, "window_size": WINDOW_SIZE}, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model + normalization + threshold to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
