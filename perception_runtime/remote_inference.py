"""Client for a separate camera-only inference server."""

import json
import socket
import struct


class RemoteYoloProcessor:
    def __init__(self, settings, debug_output_dir=None):
        self.host = str(settings.get("host", "127.0.0.1"))
        self.port = int(settings["port"])
        self.timeout_s = float(settings.get("timeout_seconds", 10.0))
        self._socket = None
        self._debug = None
        if debug_output_dir is not None:
            from .yolo_processor import YoloV5Processor
            self._debug = YoloV5Processor.__new__(YoloV5Processor)
            self._debug.camera_id = "front_rgb"
            self._debug.debug_output_dir = debug_output_dir
            from pathlib import Path
            self._debug.debug_output_dir = Path(debug_output_dir)
            (self._debug.debug_output_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self._debug.debug_output_dir / "annotated").mkdir(parents=True, exist_ok=True)
        self._cv2 = None

    def _connect(self):
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._socket.settimeout(self.timeout_s)

    def _recv_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise EOFError("inference server disconnected")
            data.extend(chunk)
        return bytes(data)

    def __call__(self, bundle):
        frame = bundle.frames["front_rgb"]
        image = frame.image_rgb
        request = {
            "episode_id": bundle.episode_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "image_width_px": frame.image_width_px,
            "image_height_px": frame.image_height_px,
            "image_rgb_bytes": int(image.nbytes),
        }
        encoded = json.dumps(request, allow_nan=False).encode("utf-8")
        self._connect()
        try:
            self._socket.sendall(struct.pack("!I", len(encoded)) + encoded + image.tobytes())
            response_size = struct.unpack("!I", self._recv_exact(4))[0]
            response = json.loads(self._recv_exact(response_size).decode("utf-8"))
        except (OSError, EOFError):
            self.close()
            raise
        for key in ("episode_id", "camera_id", "frame_id", "timestamp_ns"):
            if response.get(key) != request[key]:
                raise ValueError("remote inference response {} mismatch".format(key))
        if response.get("source") != "camera_model":
            raise ValueError("remote inference source must be camera_model")
        if self._debug is not None:
            if self._cv2 is None:
                import cv2
                self._cv2 = cv2
            self._debug._save_debug(frame.frame_id, image, response.get("detections", []))
        return {
            "camera_id": frame.camera_id,
            "coordinate_frame": "image_pixels_x_right_y_down",
            "image_width_px": frame.image_width_px,
            "image_height_px": frame.image_height_px,
            "inference_latency_ms": float(response["inference_latency_ms"]),
            "device_requested": "remote",
            "device_actual": str(response.get("device", "unknown")),
            "precision": str(response.get("precision", "fp32")),
            "detections": response.get("detections", []),
            "tracks": [],
            "tracking_enabled": False,
        }

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
