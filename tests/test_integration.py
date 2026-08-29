"""
test_integration.py - end-to-end test against a REAL Mosquitto broker
(subprocess), running the ACTUAL trained/quantized model through the
simulator, exactly as production would. Skipped if mosquitto isn't
installed, or if the model artifacts haven't been generated yet.
"""
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")
MODEL_PRESENT = os.path.exists(os.path.join(ARTIFACTS_DIR, "model_quant.tflite"))

pytestmark = [
    pytest.mark.skipif(shutil.which("mosquitto") is None, reason="mosquitto not installed"),
    pytest.mark.skipif(not MODEL_PRESENT, reason="model/artifacts/model_quant.tflite not generated yet "
                                                   "(run model/train_autoencoder.py then model/convert_tflite.py)"),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def broker(tmp_path):
    port = _free_port()
    conf_path = tmp_path / "mosquitto.conf"
    conf_path.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\nlog_type error\n")
    proc = subprocess.Popen(["mosquitto", "-c", str(conf_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    yield "127.0.0.1", port
    proc.terminate()
    proc.wait(timeout=5)


def test_sensor_readings_flow_end_to_end_into_sqlite(broker, tmp_path):
    from virtual_sensor import VirtualVibrationSensor
    from logger import MqttLogger, ReadingLogger

    host, port = broker
    db_path = str(tmp_path / "vib.sqlite3")

    mqtt_logger = MqttLogger(host, port, db_path)
    mqtt_logger.start()
    time.sleep(0.3)

    sensor = VirtualVibrationSensor(host, port, window_interval_s=0.2, fault_probability=0.0, seed=1)
    sensor.client.loop_start()
    for _ in range(5):
        sensor.step()
        time.sleep(0.2)
    sensor.client.loop_stop()
    time.sleep(0.3)
    mqtt_logger.stop()

    reader = ReadingLogger(db_path)
    assert "machine-01" in reader.devices()
    scores = reader.history("machine-01", "vibration", "anomaly_score")
    assert len(scores) >= 3
    # with fault_probability=0, every score should be well under the threshold
    assert all(float(v) < 0.05 for _, v in scores)


def test_injected_fault_produces_an_alert(broker, tmp_path):
    from virtual_sensor import VirtualVibrationSensor
    from logger import MqttLogger, ReadingLogger

    host, port = broker
    db_path = str(tmp_path / "vib.sqlite3")

    mqtt_logger = MqttLogger(host, port, db_path)
    mqtt_logger.start()
    time.sleep(0.3)

    # fault_probability=1.0 -> every window is an injected fault, so we
    # should see at least one alert=1 in the log.
    sensor = VirtualVibrationSensor(host, port, window_interval_s=0.2, fault_probability=1.0, seed=2)
    sensor.client.loop_start()
    for _ in range(3):
        sensor.step()
        time.sleep(0.2)
    sensor.client.loop_stop()
    time.sleep(0.3)
    mqtt_logger.stop()

    reader = ReadingLogger(db_path)
    alert_values = [v for _, v in reader.history("machine-01", "vibration", "alert")]
    assert "1" in alert_values

    fault_values = [v for _, v in reader.history("machine-01", "debug", "fault_type")]
    assert all(v != "none" for v in fault_values)


def test_status_reports_online_then_offline_on_clean_shutdown(broker, tmp_path):
    from virtual_sensor import VirtualVibrationSensor
    from logger import MqttLogger, ReadingLogger

    host, port = broker
    db_path = str(tmp_path / "vib.sqlite3")

    mqtt_logger = MqttLogger(host, port, db_path)
    mqtt_logger.start()
    time.sleep(0.3)

    sensor = VirtualVibrationSensor(host, port, window_interval_s=0.2, seed=3)
    sensor.client.loop_start()
    time.sleep(0.3)
    sensor.step()
    time.sleep(0.2)
    sensor.client.publish("machine/machine-01/status", "offline", retain=True)
    time.sleep(0.3)
    sensor.client.loop_stop()
    sensor.client.disconnect()
    time.sleep(0.3)
    mqtt_logger.stop()

    reader = ReadingLogger(db_path)
    status_values = [v for _, v in reader.history("machine-01", "status", "status")]
    assert "online" in status_values
    assert status_values[-1] == "offline"
