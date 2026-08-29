"""
anomaly_detector.py - loads the trained, quantized TFLite model and its
normalization/threshold parameters, and scores raw vibration windows. Used
by the simulator to run exactly the same detection logic the real ESP32
firmware runs (same model file, same math) - only the input source differs
(simulated window vs a real accelerometer read).
"""
import json
import os

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")


class AnomalyDetector:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        with open(os.path.join(artifacts_dir, "normalization.json")) as f:
            norm = json.load(f)
        self.mean = norm["mean"]
        self.std = norm["std"]
        self.window_size = norm["window_size"]

        with open(os.path.join(artifacts_dir, "threshold_quant.json")) as f:
            thresh = json.load(f)
        self.threshold = thresh["threshold"]

        with open(os.path.join(artifacts_dir, "model_quant.tflite"), "rb") as f:
            model_bytes = f.read()

        self.interpreter = Interpreter(model_content=model_bytes)
        self.interpreter.allocate_tensors()
        self._input_detail = self.interpreter.get_input_details()[0]
        self._output_detail = self.interpreter.get_output_details()[0]
        self._in_scale, self._in_zero = self._input_detail["quantization"]
        self._out_scale, self._out_zero = self._output_detail["quantization"]

    def score(self, raw_window: np.ndarray) -> tuple[float, bool]:
        """Returns (reconstruction_mse, is_anomaly)."""
        window_norm = (raw_window.astype(np.float32) - self.mean) / self.std

        q_input = np.round(window_norm / self._in_scale + self._in_zero).astype(np.int8)
        q_input = q_input.reshape(self._input_detail["shape"])
        self.interpreter.set_tensor(self._input_detail["index"], q_input)
        self.interpreter.invoke()
        q_output = self.interpreter.get_tensor(self._output_detail["index"])[0]
        recon = (q_output.astype(np.float32) - self._out_zero) * self._out_scale

        mse = float(np.mean((window_norm - recon) ** 2))
        return mse, mse > self.threshold
