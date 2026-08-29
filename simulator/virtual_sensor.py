"""
virtual_sensor.py - stands in for the real ESP32 + accelerometer: generates
vibration windows (mostly normal, with occasional injected faults), scores
each one through the SAME trained/quantized TFLite model the firmware
embeds, and publishes results over MQTT.

Unlike the esp32-smart-agriculture simulator (which re-implemented the
control logic in Python), this one literally runs the production model
file (model/artifacts/model_quant.tflite) - so what you see in the
dashboard is exactly what the on-device inference would produce, not an
approximation of it.
"""
import argparse
import os
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.dirname(__file__))

from mqtt_topics import (  # noqa: E402
    DEFAULT_DEVICE_ID, status_topic, rms_topic, anomaly_score_topic,
    alert_topic, debug_fault_type_topic,
)
from vibration_signals import generate_normal_window, generate_anomaly_window, FAULT_TYPES  # noqa: E402
from anomaly_detector import AnomalyDetector  # noqa: E402


class VirtualVibrationSensor:
    def __init__(self, broker_host: str, broker_port: int = 1883, device_id: str = DEFAULT_DEVICE_ID,
                 window_interval_s: float = 1.0, fault_probability: float = 0.15, seed: int = 0):
        self.device_id = device_id
        self.window_interval_s = window_interval_s
        self.fault_probability = fault_probability
        self.rng = np.random.default_rng(seed)
        self.detector = AnomalyDetector()
        self._last_alert = None

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"virtual-{device_id}")
        self.client.will_set(status_topic(device_id), payload="offline", retain=True)
        self.client.on_connect = self._on_connect
        self.client.connect(broker_host, broker_port)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.publish(status_topic(self.device_id), "online", retain=True)

    def step(self):
        """Generates one window, scores it, publishes results. Returns
        (rms, mse, is_anomaly, injected_fault_type_or_None) for callers
        (tests, CLI) that want the values directly."""
        injected_fault = None
        if self.rng.random() < self.fault_probability:
            injected_fault = self.rng.choice(FAULT_TYPES)
            window = generate_anomaly_window(self.rng, injected_fault)
        else:
            window = generate_normal_window(self.rng)

        rms = float(np.sqrt(np.mean(window ** 2)))
        mse, is_anomaly = self.detector.score(window)

        self.client.publish(rms_topic(self.device_id), round(rms, 4))
        self.client.publish(anomaly_score_topic(self.device_id), round(mse, 6))
        self.client.publish(debug_fault_type_topic(self.device_id), injected_fault or "none")

        if is_anomaly != self._last_alert:
            self.client.publish(alert_topic(self.device_id), "1" if is_anomaly else "0", retain=True)
            self._last_alert = is_anomaly

        return rms, mse, is_anomaly, injected_fault

    def run(self, duration_s: float | None = None):
        self.client.loop_start()
        start = time.time()
        try:
            while duration_s is None or (time.time() - start) < duration_s:
                self.step()
                time.sleep(self.window_interval_s)
        finally:
            self.client.publish(status_topic(self.device_id), "offline", retain=True)
            self.client.loop_stop()
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Virtual vibration sensor + on-device anomaly detector (MQTT)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--fault-probability", type=float, default=0.15)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    sensor = VirtualVibrationSensor(args.host, args.port, args.device_id, args.interval, args.fault_probability)
    print(f"Virtual vibration sensor '{args.device_id}' running the real TFLite model, "
          f"publishing to {args.host}:{args.port} every {args.interval}s (Ctrl+C to stop)")
    sensor.run(args.duration)


if __name__ == "__main__":
    main()
