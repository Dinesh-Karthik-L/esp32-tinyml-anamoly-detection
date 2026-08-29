"""
convert_tflite.py - converts the trained Keras autoencoder to a fully
int8-quantized TFLite model (the format TensorFlow Lite for Microcontrollers
needs), then re-runs the anomaly-detection evaluation THROUGH the quantized
model to confirm quantization didn't wreck detection accuracy - this is
the step that actually validates the model is fit to deploy, not just that
conversion succeeded without an exception.

Saves to model/artifacts/:
    model_quant.tflite       - the quantized model
    threshold_quant.json     - threshold + eval metrics recomputed on the
                                quantized model's (dequantized) outputs

Usage:
    python model/convert_tflite.py
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from vibration_signals import WINDOW_SIZE  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def convert(model_path: str, representative_windows: np.ndarray) -> bytes:
    model = tf.keras.models.load_model(model_path)

    def representative_dataset():
        for i in range(min(200, len(representative_windows))):
            yield [representative_windows[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def run_quantized_inference(tflite_model: bytes, windows_norm: np.ndarray) -> np.ndarray:
    """Runs each window through the quantized model and returns the
    reconstruction MSE computed in dequantized float space (i.e. the same
    units the float model's threshold was computed in)."""
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]

    errors = np.zeros(len(windows_norm), dtype=np.float32)
    for i, window in enumerate(windows_norm):
        q_input = np.round(window / in_scale + in_zero).astype(np.int8).reshape(input_detail["shape"])
        interpreter.set_tensor(input_detail["index"], q_input)
        interpreter.invoke()
        q_output = interpreter.get_tensor(output_detail["index"])[0]
        recon = (q_output.astype(np.float32) - out_zero) * out_scale
        errors[i] = np.mean((window - recon) ** 2)
    return errors


def main():
    parser = argparse.ArgumentParser(description="Convert the trained autoencoder to a quantized TFLite model")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz"))
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    args = parser.parse_args()

    with open(os.path.join(ARTIFACTS_DIR, "normalization.json")) as f:
        norm = json.load(f)
    mean, std = norm["mean"], norm["std"]

    data = np.load(args.dataset, allow_pickle=True)
    normal, anomalies = data["normal"], data["anomalies"]
    normal_norm = (normal.astype(np.float32) - mean) / std
    anomalies_norm = (anomalies.astype(np.float32) - mean) / std

    model_path = os.path.join(ARTIFACTS_DIR, "autoencoder.keras")
    tflite_model = convert(model_path, normal_norm)

    out_path = os.path.join(ARTIFACTS_DIR, "model_quant.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    # Re-run detection metrics THROUGH the quantized model.
    normal_err = run_quantized_inference(tflite_model, normal_norm)
    anomaly_err = run_quantized_inference(tflite_model, anomalies_norm)
    threshold = float(np.percentile(normal_err, args.threshold_percentile))

    metrics = {
        "model_size_bytes": len(tflite_model),
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "normal_error_mean": float(normal_err.mean()),
        "anomaly_error_mean": float(anomaly_err.mean()),
        "false_positive_rate": float(np.mean(normal_err > threshold)),
        "recall": float(np.mean(anomaly_err > threshold)),
    }

    with open(os.path.join(ARTIFACTS_DIR, "threshold_quant.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nQuantized model ({len(tflite_model)} bytes) saved to {out_path}")


if __name__ == "__main__":
    main()
