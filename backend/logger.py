"""
logger.py - subscribes to vibration/status/debug topics and persists every
reading to SQLite, same pattern as esp32-smart-agriculture/backend/logger.py.
"""
import argparse
import sqlite3
import time

import paho.mqtt.client as mqtt

from mqtt_topics import VIBRATION_WILDCARD, STATUS_WILDCARD, DEBUG_WILDCARD

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    device_id TEXT NOT NULL,
    category TEXT NOT NULL,   -- 'vibration' | 'status' | 'debug'
    metric TEXT NOT NULL,     -- 'rms' | 'anomaly_score' | 'alert' | 'status' | 'fault_type'
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_readings_metric ON readings(device_id, category, metric);
"""


def parse_topic(topic: str):
    """machine/<device_id>/vibration/<metric>  -> ('vibration', device_id, metric)
       machine/<device_id>/status               -> ('status', device_id, 'status')
       machine/<device_id>/debug/<metric>        -> ('debug', device_id, metric)
    """
    parts = topic.split("/")
    if len(parts) == 4 and parts[0] == "machine" and parts[2] == "vibration":
        return "vibration", parts[1], parts[3]
    if len(parts) == 4 and parts[0] == "machine" and parts[2] == "debug":
        return "debug", parts[1], parts[3]
    if len(parts) == 3 and parts[0] == "machine" and parts[2] == "status":
        return "status", parts[1], "status"
    return None


class ReadingLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(self, topic: str, payload: str, timestamp: float | None = None) -> bool:
        parsed = parse_topic(topic)
        if parsed is None:
            return False
        category, device_id, metric = parsed
        ts = timestamp if timestamp is not None else time.time()
        self.conn.execute(
            "INSERT INTO readings (timestamp, device_id, category, metric, value) VALUES (?, ?, ?, ?, ?)",
            (ts, device_id, category, metric, payload),
        )
        self.conn.commit()
        return True

    def latest(self, device_id: str, category: str, metric: str):
        return self.conn.execute(
            "SELECT timestamp, value FROM readings WHERE device_id=? AND category=? AND metric=? "
            "ORDER BY timestamp DESC LIMIT 1",
            (device_id, category, metric),
        ).fetchone()

    def history(self, device_id: str, category: str, metric: str, since: float = 0.0, limit: int = 2000):
        return self.conn.execute(
            "SELECT timestamp, value FROM readings WHERE device_id=? AND category=? AND metric=? "
            "AND timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (device_id, category, metric, since, limit),
        ).fetchall()

    def devices(self):
        return [r[0] for r in self.conn.execute("SELECT DISTINCT device_id FROM readings").fetchall()]

    def close(self):
        self.conn.close()


class MqttLogger:
    def __init__(self, broker_host: str, broker_port: int, db_path: str):
        self.reading_logger = ReadingLogger(db_path)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vibration-logger")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker_host, broker_port)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe(VIBRATION_WILDCARD)
        client.subscribe(STATUS_WILDCARD)
        client.subscribe(DEBUG_WILDCARD)

    def _on_message(self, client, userdata, msg):
        self.reading_logger.record(msg.topic, msg.payload.decode(errors="ignore"))

    def start(self):
        self.client.loop_start()

    def run_forever(self):
        self.client.loop_forever()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.reading_logger.close()


def main():
    parser = argparse.ArgumentParser(description="MQTT -> SQLite logger for the vibration anomaly detector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--db", default="vibration_data.sqlite3")
    args = parser.parse_args()

    logger = MqttLogger(args.host, args.port, args.db)
    print(f"Logging {args.host}:{args.port} -> {args.db} (Ctrl+C to stop)")
    try:
        logger.run_forever()
    except KeyboardInterrupt:
        logger.stop()


if __name__ == "__main__":
    main()
