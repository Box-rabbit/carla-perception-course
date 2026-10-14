"""Run a reproducible single-image YOLOv5 interface and visual check."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPOSITORY_ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from perception_detector import YoloV5Detector  # noqa: E402


def _draw(image_bgr, detections):
    output = image_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = item["bbox2d_xyxy_px"]
        color = (0, 200, 0) if item["class"] == "vehicle" else (0, 0, 255)
        label = "{} {:.2f}".format(item["subclass"], item["confidence"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPOSITORY_ROOT / "test_data/images/image.png")
    parser.add_argument("--weights", type=Path, default=YOLO_ROOT / "yolov5s.pt")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs/yolo_validation/offline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 on CUDA; ignored on CPU")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    image_bgr = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise IOError("cannot read image: {}".format(args.source))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detector = YoloV5Detector(weights=args.weights, device=args.device, half=args.half)
    records = []
    first = None
    for index in range(args.iterations):
        started = time.perf_counter()
        detections = detector.detect(image_rgb)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if first is None:
            first = detections
        records.append({"iteration": index, "latency_ms": latency_ms,
                        "detection_count": len(detections)})
    if not isinstance(first, list) or any("truth" in str(item).lower() for item in first):
        raise AssertionError("invalid detector JSON interface")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw.png"
    annotated_path = output / "annotated.png"
    if not cv2.imwrite(str(raw_path), image_bgr):
        raise IOError("failed to save raw image: {}".format(raw_path))
    if not cv2.imwrite(str(annotated_path), _draw(image_bgr, first)):
        raise IOError("failed to save annotated image: {}".format(annotated_path))
    report = {
        "source": str(args.source.resolve()),
        "weights": str(args.weights.resolve()),
        "device_requested": str(args.device),
        "device_actual": str(detector.device),
        "precision": "fp16" if detector.use_half else "fp32",
        "image_width_px": int(image_rgb.shape[1]),
        "image_height_px": int(image_rgb.shape[0]),
        "json_interface": {"valid": True, "detection_count": len(first)},
        "visual_outputs": {"raw": str(raw_path.resolve()), "annotated": str(annotated_path.resolve())},
        "latency_ms": {
            "mean": sum(item["latency_ms"] for item in records) / len(records),
            "min": min(item["latency_ms"] for item in records),
            "max": max(item["latency_ms"] for item in records),
            "iterations": records,
        },
        "detections": first,
    }
    with (output / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
