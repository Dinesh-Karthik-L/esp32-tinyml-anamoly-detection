import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from logger import ReadingLogger, parse_topic  # noqa: E402


def test_parse_topic_vibration():
    assert parse_topic("machine/machine-01/vibration/anomaly_score") == ("vibration", "machine-01", "anomaly_score")


def test_parse_topic_status():
    assert parse_topic("machine/machine-01/status") == ("status", "machine-01", "status")


def test_parse_topic_debug():
    assert parse_topic("machine/machine-01/debug/fault_type") == ("debug", "machine-01", "fault_type")


def test_parse_topic_rejects_unrelated():
    assert parse_topic("some/other/topic") is None


def test_record_and_latest_roundtrip(tmp_path):
    logger = ReadingLogger(str(tmp_path / "t.sqlite3"))
    assert logger.record("machine/machine-01/vibration/anomaly_score", "0.042", timestamp=10.0) is True
    assert logger.latest("machine-01", "vibration", "anomaly_score") == (10.0, "0.042")


def test_history_ordered_and_filtered_by_since(tmp_path):
    logger = ReadingLogger(str(tmp_path / "t.sqlite3"))
    for ts, val in [(1.0, "0.01"), (2.0, "0.02"), (3.0, "0.9")]:
        logger.record("machine/machine-01/vibration/anomaly_score", val, timestamp=ts)
    rows = logger.history("machine-01", "vibration", "anomaly_score", since=1.5)
    assert [r[1] for r in rows] == ["0.02", "0.9"]


def test_devices_lists_distinct_ids(tmp_path):
    logger = ReadingLogger(str(tmp_path / "t.sqlite3"))
    logger.record("machine/m1/status", "online", timestamp=1.0)
    logger.record("machine/m2/status", "online", timestamp=1.0)
    assert sorted(logger.devices()) == ["m1", "m2"]
