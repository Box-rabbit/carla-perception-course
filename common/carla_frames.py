"""Shared conversion of CARLA camera callback buffers to RGB arrays."""

from numbers import Integral

import numpy as np


def carla_image_to_rgb(image, expected_size=None):
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("invalid CARLA image dimensions")
    if expected_size is not None and (int(width), int(height)) != tuple(expected_size):
        raise ValueError(
            f"expected CARLA image size {tuple(expected_size)}, got {(width, height)}"
        )
    raw_data = getattr(image, "raw_data", None)
    if raw_data is None:
        raise ValueError("CARLA image has no raw_data buffer")
    bgra = np.frombuffer(raw_data, dtype=np.uint8)
    if bgra.size != int(width) * int(height) * 4:
        raise ValueError("CARLA BGRA buffer size does not match image dimensions")
    bgra = bgra.reshape((int(height), int(width), 4))
    return np.ascontiguousarray(bgra[:, :, 2::-1])

