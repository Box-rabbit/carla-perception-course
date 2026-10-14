from pathlib import Path

import numpy as np


TRAFFIC_LIGHT_CLASS_ID = 9
TRAFFIC_LIGHT_CLASS_NAME = "traffic light"


def _format_detections(rows, image_height, image_width):
    detections = []
    for row in rows:
        if int(row[5]) != TRAFFIC_LIGHT_CLASS_ID:
            continue
        coordinates = np.asarray(row[:4], dtype=np.float64)
        confidence = float(row[4])
        if not np.isfinite(coordinates).all() or not np.isfinite(confidence):
            continue
        coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, image_width)
        coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, image_height)
        coordinates = np.rint(coordinates).astype(int).tolist()
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        detections.append({
            "class_id": TRAFFIC_LIGHT_CLASS_ID,
            "class": "traffic_light",
            "bbox2d_xyxy_px": coordinates,
            "confidence": confidence,
        })
    return detections


class TrafficLightDetector:
    def __init__(
        self,
        yolo_dir,
        weights=None,
        image_size=640,
        device="0",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections=300,
    ):
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("imgsz must be a positive integer")
        if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("conf-thres and iou-thres must be in [0, 1]")
        if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections <= 0:
            raise ValueError("max_detections must be a positive integer")
        self.yolo_dir = Path(yolo_dir).expanduser().resolve()
        if not self.yolo_dir.is_dir():
            raise FileNotFoundError(f"YOLO directory not found: {self.yolo_dir}")
        self.weights_path = (
            Path(weights).expanduser().resolve()
            if weights is not None
            else self.yolo_dir / "yolov5s.pt"
        )
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        if self.weights_path.suffix.lower() != ".pt":
            raise ValueError("This interface expects YOLOv5 .pt weights")

        import sys
        if str(self.yolo_dir) not in sys.path:
            sys.path.insert(0, str(self.yolo_dir))
        import torch
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device

        self._torch = torch
        self._letterbox = letterbox
        self._non_max_suppression = non_max_suppression
        self._scale_coords = scale_coords
        self.device = select_device(device)
        self.model = DetectMultiBackend(str(self.weights_path), device=self.device, fp16=False)
        if self.model.names[TRAFFIC_LIGHT_CLASS_ID] != TRAFFIC_LIGHT_CLASS_NAME:
            raise ValueError("Weights do not use the expected COCO traffic-light class")
        self.image_size = check_img_size(image_size, s=self.model.stride)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.model.warmup(imgsz=(1, 3, self.image_size, self.image_size))

    @staticmethod
    def _validate_rgb(image_rgb):
        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
            raise TypeError("Input must be a uint8 NumPy array in RGB order")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input shape must be H x W x 3")
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            raise ValueError("Input image must not be empty")

    def detect(self, image_rgb):
        self._validate_rgb(image_rgb)
        image_height, image_width = image_rgb.shape[:2]
        with self._torch.inference_mode():
            padded, ratio, padding = self._letterbox(
                image_rgb, self.image_size, stride=self.model.stride, auto=True
            )
            channels_first = np.ascontiguousarray(padded.transpose((2, 0, 1)))
            tensor = self._torch.from_numpy(channels_first).to(self.device).float()
            tensor = (tensor / 255.0).unsqueeze(0)
            predictions = self.model(tensor)
            filtered = self._non_max_suppression(
                predictions,
                conf_thres=self.confidence_threshold,
                iou_thres=self.iou_threshold,
                classes=[TRAFFIC_LIGHT_CLASS_ID],
                max_det=self.max_detections,
            )[0]
            if len(filtered) == 0:
                return []
            filtered[:, :4] = self._scale_coords(
                tensor.shape[2:], filtered[:, :4], image_rgb.shape[:2],
                ratio_pad=(ratio, padding),
            ).round()
            rows = filtered.detach().cpu().numpy()
        return _format_detections(rows, image_height, image_width)
