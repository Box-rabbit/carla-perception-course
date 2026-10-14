"""Compatibility imports for the canonical shared config package."""

from common.config import (  # noqa: F401
    DEFAULT_SCENARIO_CONFIG,
    REPOSITORY_ROOT,
    enabled_sensors,
    load_runtime_config,
    load_scenario_config,
    load_sensor_config,
    load_perception_config,
    load_system_config,
    validate_python_runtime,
    validate_sensor_contract,
)

__all__ = [
    "DEFAULT_SCENARIO_CONFIG",
    "REPOSITORY_ROOT",
    "enabled_sensors",
    "load_runtime_config",
    "load_scenario_config",
    "load_sensor_config",
    "load_perception_config",
    "load_system_config",
    "validate_python_runtime",
    "validate_sensor_contract",
]
