"""Separate JSONL logs for model output and simulator truth."""

import json
from pathlib import Path

from .runtime import _find_forbidden_key
from common.contracts import (
    FRAME_ID_SOURCE,
    GROUND_TRUTH_SOURCE,
    TIMESTAMP_SOURCE,
    TRUTH_FRAME_ID_SOURCE,
    TRUTH_SCHEMA_VERSION,
)


class EpisodeLogger:
    """Write perception and truth records to different files.

    The two methods intentionally have separate signatures, preventing a
    caller from accidentally passing scenario truth into perception output.
    """

    def __init__(self, output_dir, perception_name="perception.jsonl", truth_name="truth.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._perception = (self.output_dir / perception_name).open(
            "x", encoding="utf-8"
        )
        self._truth = (self.output_dir / truth_name).open(
            "x", encoding="utf-8"
        )

    @staticmethod
    def _write(stream, record):
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()

    def log_perception(self, result):
        forbidden = _find_forbidden_key(result)
        if forbidden:
            raise ValueError(f"perception record contains truth field: {forbidden}")
        self._write(self._perception, result)

    def log_truth(self, episode_id, frame_id, timestamp_ns, truth):
        record = {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "episode_id": episode_id,
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "timestamp_source": TIMESTAMP_SOURCE,
            "frame_id_source": TRUTH_FRAME_ID_SOURCE,
            "source": GROUND_TRUTH_SOURCE,
            "truth": truth,
        }
        self._write(self._truth, record)

    def close(self):
        self._perception.close()
        self._truth.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
