"""
mqtt_topics.py - MQTT topic scheme for the vibration anomaly detector,
mirroring the style used in the esp32-smart-agriculture project.

    machine/<device_id>/status                retained "online"/"offline" (LWT)
    machine/<device_id>/vibration/rms          root-mean-square amplitude of the raw window
    machine/<device_id>/vibration/anomaly_score  reconstruction MSE from the autoencoder
    machine/<device_id>/vibration/alert        retained "0"/"1" - score > threshold
    machine/<device_id>/debug/fault_type       simulator-only: ground truth being injected,
                                                 for demo purposes (the real firmware has no
                                                 such topic - it doesn't know the ground truth)
"""

DEFAULT_DEVICE_ID = "machine-01"


def status_topic(device_id: str = DEFAULT_DEVICE_ID) -> str:
    return f"machine/{device_id}/status"


def rms_topic(device_id: str = DEFAULT_DEVICE_ID) -> str:
    return f"machine/{device_id}/vibration/rms"


def anomaly_score_topic(device_id: str = DEFAULT_DEVICE_ID) -> str:
    return f"machine/{device_id}/vibration/anomaly_score"


def alert_topic(device_id: str = DEFAULT_DEVICE_ID) -> str:
    return f"machine/{device_id}/vibration/alert"


def debug_fault_type_topic(device_id: str = DEFAULT_DEVICE_ID) -> str:
    return f"machine/{device_id}/debug/fault_type"


VIBRATION_WILDCARD = "machine/+/vibration/#"
STATUS_WILDCARD = "machine/+/status"
DEBUG_WILDCARD = "machine/+/debug/#"
