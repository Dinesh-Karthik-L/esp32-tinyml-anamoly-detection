"""
generate_dataset.py - generates and saves the training/evaluation dataset
to data/dataset.npz using simulator/vibration_signals.py.

Usage:
    python data/generate_dataset.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from vibration_signals import generate_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate the vibration anomaly-detection dataset")
    parser.add_argument("--n-normal", type=int, default=2000)
    parser.add_argument("--n-anomaly-per-type", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset.npz"))
    args = parser.parse_args()

    normal, anomalies, labels = generate_dataset(args.n_normal, args.n_anomaly_per_type, args.seed)
    np.savez(args.out, normal=normal, anomalies=anomalies, labels=labels)
    print(f"Saved {len(normal)} normal windows and {len(anomalies)} anomaly windows to {args.out}")
    print(f"Anomaly types: {sorted(set(labels))}")


if __name__ == "__main__":
    main()
