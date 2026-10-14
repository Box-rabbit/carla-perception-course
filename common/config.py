"""Canonical project configuration loader."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

from .contracts import (
    FRAME_ID_SOURCE,
    GROUND_TRUTH_SOURCE,
    IMAGE_COLOR_ORDER,
    IMAGE_COORDINATE_FRAME,
    TIMESTAMP_SOURCE,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = REPOSITORY_ROOT / "config" / "system.yaml"
SENSOR_CONFIG = REPOSITORY_ROOT / "config" / "sensors.yaml"
PERCEPTION_CONFIG = REPOSITORY_ROOT / "config" / "perception.yaml"
DEFAULT_SCENARIO_CONFIG = REPOSITORY_ROOT / "My_scenario" / "configs" / "my_scenario.yaml"


def _load_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_system_config(path=SYSTEM_CONFIG):
    return _load_yaml(path)


def load_sensor_config(path=SENSOR_CONFIG):
    config = _load_yaml(path)
    if not isinstance(config.get("sensors"), dict) or not config["sensors"]:
        raise ValueError("sensor config must contain a nonempty sensors mapping")
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("sensor defaults must be a mapping")
    resolved = {}
    for camera_id, spec in config["sensors"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"sensor spec must be a mapping: {camera_id}")
        merged = deepcopy(defaults)
        merged.update(spec)
        transform = deepcopy(defaults.get("transform", {}))
        transform.update(spec.get("transform", {}))
        merged["transform"] = transform
        resolved[camera_id] = merged
    config["sensors"] = resolved
    return config


def load_perception_config(path=PERCEPTION_CONFIG):
    config = _load_yaml(path)
    yolo = config.get("yolov5")
    if not isinstance(yolo, dict):
        raise ValueError("perception config must contain a yolov5 mapping")
    if not isinstance(yolo.get("enabled"), bool):
        raise ValueError("perception.yolov5.enabled must be boolean")
    weights = Path(yolo.get("weights", ""))
    if not weights.is_absolute():
        weights = REPOSITORY_ROOT / weights
    yolo["weights"] = str(weights.resolve())
    return config


def load_scenario_config(path=DEFAULT_SCENARIO_CONFIG):
    return _load_yaml(path)


def load_runtime_config(
    scenario_path=DEFAULT_SCENARIO_CONFIG,
    system_path=SYSTEM_CONFIG,
    sensor_path=SENSOR_CONFIG,
):
    """Load all project config while keeping scenario fields at the top level."""
    scenario = load_scenario_config(scenario_path)
    system = load_system_config(system_path)
    sensors = load_sensor_config(sensor_path)
    perception = load_perception_config()
    merged = deepcopy(scenario)
    scenario_root = Path(scenario_path).expanduser().resolve().parent.parent
    route_file = merged.get("route", {}).get("route_file")
    if route_file:
        merged["route"]["route_file"] = str(
            (scenario_root / route_file).resolve()
        )
    merged["_system"] = system
    merged["_sensors"] = sensors
    merged["_perception"] = perception
    return merged


def validate_python_runtime(system_config):
    """Require the Python version supported by the CARLA 0.9.15 API."""
    expected = str(system_config.get("project", {}).get("python_version", "3.7"))
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected:
        raise RuntimeError(
            f"project requires Python {expected}; current interpreter is Python {actual}"
        )
    return True


def enabled_sensors(sensor_config):
    return {
        camera_id: spec
        for camera_id, spec in sensor_config["sensors"].items()
        if bool(spec.get("enabled", False))
    }


def validate_sensor_contract(system_config, sensor_config):
    contract = system_config.get("data_contract", {})
    expected_contract = {
        "frame_id_source": FRAME_ID_SOURCE,
        "timestamp_source": TIMESTAMP_SOURCE,
        "image_color_order": IMAGE_COLOR_ORDER,
        "image_coordinate_frame": IMAGE_COORDINATE_FRAME,
        "truth_source": GROUND_TRUTH_SOURCE,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"data_contract.{key} must be {expected!r}")
    runtime = system_config.get("runtime", {})
    fixed_delta = float(runtime["fixed_delta_seconds"])
    sensor_tick = float(runtime["sensor_tick_seconds"])
    if fixed_delta <= 0 or sensor_tick <= 0:
        raise ValueError("fixed_delta_seconds and sensor_tick_seconds must be positive")
    if abs(fixed_delta - sensor_tick) > 1e-12:
        raise ValueError("world fixed delta and sensor tick must match")
    for camera_id, spec in sensor_config["sensors"].items():
        size = spec.get("image_size_px")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
        ):
            raise ValueError(f"invalid image_size_px for {camera_id}")
        if "sensor_tick_seconds" in spec:
            raise ValueError(
                f"sensor tick is global in system.yaml; remove per-sensor value for {camera_id}"
            )
        if float(spec.get("fov_deg", 0.0)) <= 0:
            raise ValueError(f"fov_deg must be positive for {camera_id}")
        if spec.get("enabled") and spec.get("calibration_file"):
            calibration_path = (REPOSITORY_ROOT / spec["calibration_file"]).resolve()
            if not calibration_path.is_file():
                raise ValueError(f"calibration file missing for {camera_id}: {calibration_path}")
            with calibration_path.open(encoding="utf-8") as stream:
                calibration = json.load(stream)
            if calibration.get("camera_id") != camera_id:
                raise ValueError(f"calibration camera_id mismatch for {camera_id}")
            if calibration.get("image_size_px") != size:
                raise ValueError(f"calibration image size mismatch for {camera_id}")
    return True
