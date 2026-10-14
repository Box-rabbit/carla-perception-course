import argparse
import json
from pathlib import Path
import time

import cv2

from model_assets import DEFAULT_MODEL_DIR
from traffic_light_classifier import TrafficLightClassifier


def source_images(source):
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        paths = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if paths:
            return paths
    raise ValueError(f"No readable image files found: {source}")


def classify_image(classifier, path, bbox=None):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = bgr.shape[:2]
    if bbox is None:
        bbox = (0, 0, width, height)
    left, top, right, bottom = bbox
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"ROI is outside original image dimensions {width} x {height}")
    rgb_roi = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    prediction = classifier.classify(rgb_roi)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "schema_version": "traffic-light-roi-0.1",
        "model": "autoware_mobilenetv2_v2_batch1",
        "backend": "opencv_dnn_cpu",
        "source": str(path.resolve()),
        "source_image_width_px": width,
        "source_image_height_px": height,
        "roi_xyxy_px": list(bbox),
        "roi_width_px": right - left,
        "roi_height_px": bottom - top,
        "classification_call_ms": elapsed_ms,
        "status": "ok",
        "prediction_valid": True,
        **prediction,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a single unannotated signal-head ROI, or a folder of ROIs")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error(f"Output already exists; use a new filename: {arguments.output}")
    paths = source_images(arguments.source)
    if arguments.roi is not None and not arguments.source.is_file():
        parser.error("--roi requires one original image file")
    classifier = TrafficLightClassifier(arguments.model_dir, arguments.min_confidence)
    records = []
    for path in paths:
        try:
            record = classify_image(classifier, path, arguments.roi)
        except (ValueError, cv2.error) as error:
            record = {
                "schema_version": "traffic-light-roi-0.1",
                "source": str(path.resolve()),
                "status": "error",
                "prediction_valid": False,
                "state": "unknown",
                "error": str(error),
            }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "traffic-light-roi-batch-0.1", "results": records}, stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"JSON saved: {arguments.output.resolve()}")
    if any(record["status"] == "error" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
