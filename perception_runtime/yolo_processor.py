"""Project-owned YOLOv5 processor for the synchronized RGB frame contract."""

import sys
import time
from pathlib import Path

import numpy as np

from .frame_schema import SensorFrameBundle
from .tracker import IoUTracker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPOSITORY_ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from perception_detector import YoloV5Detector  # noqa: E402


class YoloV5Processor:
    """Run one detector on ``front_rgb`` and optionally save debug images.

    The processor only receives a ``SensorFrameBundle``.  It never receives
    CARLA actors or scenario state, so truth cannot enter model inference.
    """

    def __init__(self, settings, debug_output_dir=None):
        self.camera_id = "front_rgb"
        self.detector = YoloV5Detector(
            weights=settings["weights"],
            image_size=int(settings.get("image_size", 640)),
            device=settings.get("device", "auto"),
            half=bool(settings.get("half", False)),
            confidence_threshold=float(settings.get("confidence_threshold", 0.25)),
            iou_threshold=float(settings.get("iou_threshold", 0.45)),
            max_detections=int(settings.get("max_detections", 300)),
        )
        tracking = settings.get("tracking", {})
        self._tracker = IoUTracker(
            iou_threshold=float(tracking.get("iou_threshold", 0.3)),
            max_missed_frames=int(tracking.get("max_missed_frames", 3)),
        ) if bool(tracking.get("enabled", True)) else None
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_output_dir is not None:
            (self.debug_output_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.debug_output_dir / "annotated").mkdir(parents=True, exist_ok=True)

    def __call__(self, bundle):
        if not isinstance(bundle, SensorFrameBundle):
            raise TypeError("YoloV5Processor expects SensorFrameBundle")
        if self.camera_id not in bundle.frames:
            raise KeyError("required camera is missing: front_rgb")
        frame = bundle.frames[self.camera_id]
        image_rgb = frame.image_rgb
        started = time.perf_counter()
        detections = self.detector.detect(image_rgb)
        inference_latency_ms = (time.perf_counter() - started) * 1000.0
        tracker = getattr(self, "_tracker", None)
        tracks = tracker.update(detections) if tracker is not None else []
        if self.debug_output_dir is not None:
            self._save_debug(frame.frame_id, image_rgb, detections)
        return {
            "camera_id": self.camera_id,
            "coordinate_frame": "image_pixels_x_right_y_down",
            "image_width_px": frame.image_width_px,
            "image_height_px": frame.image_height_px,
            "inference_latency_ms": inference_latency_ms,
            "device_requested": getattr(self.detector, "device_requested", "unknown"),
            "device_actual": str(getattr(self.detector, "device", "unknown")),
            "precision": "fp16" if bool(getattr(self.detector, "use_half", False)) else "fp32",
            "detections": detections,
            "tracks": tracks,
            "tracking_enabled": tracker is not None,
        }

    def _save_debug(self, frame_id, image_rgb, detections):
        import cv2

        raw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        annotated = raw.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox2d_xyxy_px"]
            color = (0, 200, 0) if detection["class"] == "vehicle" else (0, 0, 255)
            label = "{} {:.2f}".format(detection["subclass"], detection["confidence"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        stem = "frame_{:010d}.png".format(int(frame_id))
        if not cv2.imwrite(str(self.debug_output_dir / "raw" / stem), raw):
            raise IOError("failed to write raw debug image")
        if not cv2.imwrite(str(self.debug_output_dir / "annotated" / stem), annotated):
            raise IOError("failed to write annotated debug image")


def build_yolo_processor(config, debug_output_dir=None):
    settings = config.get("yolov5", config)
    if str(config.get("inference_backend", "local")).lower() == "remote":
        from .remote_inference import RemoteYoloProcessor
        return RemoteYoloProcessor(config.get("remote_inference", {}),
                                   debug_output_dir=debug_output_dir)
    return YoloV5Processor(settings, debug_output_dir=debug_output_dir)
