"""Online perception processor registry with truth isolation."""

import time

from common.contracts import (
    CAMERA_MODEL_SOURCE,
    FRAME_ID_SOURCE,
    FRAME_SCHEMA_VERSION,
    TIMESTAMP_SOURCE,
)


_FORBIDDEN_TRUTH_KEYS = {
    "scenario_observation",
    "traffic_light_ground_truth",
    "ground_truth",
    "truth",
    "actor_state",
}


def _find_forbidden_key(value, path="result"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_TRUTH_KEYS:
                return f"{path}.{key}"
            found = _find_forbidden_key(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


class PerceptionRuntime:
    """Run registered perception processors on synchronized sensor frames.

    A processor receives only ``SensorFrameBundle``.  It cannot receive the
    scenario state through this API, which keeps ground truth out of inference.
    """

    def __init__(self, processors=None):
        self._processors = {}
        for name, processor in (processors or {}).items():
            self.register(name, processor)

    def register(self, name, processor):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("processor name must be nonempty")
        if not callable(processor):
            raise TypeError("processor must be callable")
        if name in self._processors:
            raise RuntimeError(f"processor already registered: {name}")
        self._processors[name] = processor

    def close(self):
        for processor in self._processors.values():
            close = getattr(processor, "close", None)
            if callable(close):
                close()

    def process(self, bundle):
        from .frame_schema import SensorFrameBundle

        if not isinstance(bundle, SensorFrameBundle):
            raise TypeError("process expects SensorFrameBundle")
        started = time.perf_counter()
        results = {}
        errors = []
        for name, processor in self._processors.items():
            processor_started = time.perf_counter()
            try:
                output = processor(bundle)
                forbidden = _find_forbidden_key(output)
                if forbidden:
                    raise ValueError(
                        f"processor output contains forbidden truth field: {forbidden}"
                    )
                results[name] = {
                    "source": CAMERA_MODEL_SOURCE,
                    "status": "ok",
                    "result": output,
                    "latency_ms": (time.perf_counter() - processor_started) * 1000.0,
                }
            except Exception as error:  # keep the frame stream alive
                errors.append({
                    "processor": name,
                    "code": "processor_failed",
                    "message": f"{type(error).__name__}: {error}",
                })
                results[name] = {
                    "source": CAMERA_MODEL_SOURCE,
                    "status": "error",
                    "result": None,
                    "latency_ms": (time.perf_counter() - processor_started) * 1000.0,
                }
        return {
            "schema_version": FRAME_SCHEMA_VERSION,
            "episode_id": bundle.episode_id,
            "frame_id": bundle.frame_id,
            "timestamp_ns": bundle.timestamp_ns,
            "timestamp_source": TIMESTAMP_SOURCE,
            "frame_id_source": FRAME_ID_SOURCE,
            "source": CAMERA_MODEL_SOURCE if self._processors else "sensor_only",
            "camera_ids": sorted(bundle.frames),
            "frames": {
                camera_id: frame.metadata()
                for camera_id, frame in bundle.frames.items()
            },
            "status": "ok" if not errors else "partial",
            "results": results,
            "errors": errors,
            "timing": {
                "total_latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        }
