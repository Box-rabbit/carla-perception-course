"""Frame contracts shared by the CARLA bridge and perception processors.

The module intentionally contains no scenario or actor truth fields.  Ground
truth is recorded by :mod:`perception_runtime.logger` on a separate channel.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from numbers import Integral, Real
import math

import numpy as np

from common.carla_frames import carla_image_to_rgb
from common.contracts import (
    FRAME_ID_SOURCE,
    IMAGE_COLOR_ORDER,
    IMAGE_COORDINATE_FRAME,
    TIMESTAMP_SOURCE,
)


def seconds_to_nanoseconds(seconds):
    """Convert a CARLA simulation timestamp without binary float drift."""
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise ValueError("timestamp seconds must be a finite nonnegative number")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timestamp seconds must be a finite nonnegative number")
    return int(
        (Decimal(str(seconds)) * 1000000000).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def _require_nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_rgb(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("image_rgb must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image_rgb must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape H x W x 3")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image_rgb must be nonempty")


@dataclass(frozen=True)
class SensorFrame:
    """One RGB image and its simulation metadata."""

    episode_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    image_rgb: object

    def __post_init__(self):
        _require_nonempty_string(self.episode_id, "episode_id")
        _require_nonempty_string(self.camera_id, "camera_id")
        for name in ("frame_id", "timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        _validate_rgb(self.image_rgb)

    @property
    def image_width_px(self):
        return int(self.image_rgb.shape[1])

    @property
    def image_height_px(self):
        return int(self.image_rgb.shape[0])

    def metadata(self):
        return {
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_source": TIMESTAMP_SOURCE,
            "frame_id_source": FRAME_ID_SOURCE,
            "image_color_order": IMAGE_COLOR_ORDER,
            "coordinate_frame": IMAGE_COORDINATE_FRAME,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
        }


@dataclass(frozen=True)
class SensorFrameBundle:
    """Synchronized frames from one simulation tick."""

    episode_id: str
    frame_id: int
    timestamp_ns: int
    frames: dict

    def __post_init__(self):
        _require_nonempty_string(self.episode_id, "episode_id")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, Integral):
            raise ValueError("frame_id must be a nonnegative integer")
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, Integral):
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if self.frame_id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame_id and timestamp_ns must be nonnegative")
        if not isinstance(self.frames, dict) or not self.frames:
            raise ValueError("frames must be a nonempty dictionary")
        for camera_id, frame in self.frames.items():
            if not isinstance(camera_id, str) or not camera_id.strip():
                raise ValueError("camera IDs must be nonempty strings")
            if not isinstance(frame, SensorFrame):
                raise ValueError("frames must contain SensorFrame values")
            if frame.episode_id != self.episode_id:
                raise ValueError("all frames must belong to the same episode")
            if frame.frame_id != self.frame_id or frame.timestamp_ns != self.timestamp_ns:
                raise ValueError("all frames in a bundle must have identical timing")

    def metadata(self):
        return {
            "episode_id": self.episode_id,
            "frame_id": int(self.frame_id),
            "timestamp_ns": int(self.timestamp_ns),
            "camera_ids": sorted(self.frames),
            "timestamp_source": TIMESTAMP_SOURCE,
            "frame_id_source": FRAME_ID_SOURCE,
        }


def frame_from_carla_image(image, episode_id, camera_id, expected_size=None):
    """Convert a CARLA BGRA image into a validated :class:`SensorFrame`."""
    rgb = carla_image_to_rgb(image, expected_size=expected_size)
    return SensorFrame(
        episode_id=episode_id,
        camera_id=camera_id,
        frame_id=int(image.frame),
        timestamp_ns=seconds_to_nanoseconds(image.timestamp),
        image_rgb=rgb,
    )
