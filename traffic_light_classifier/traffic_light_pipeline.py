import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_YOLO_DIR = PROJECT_ROOT / "yolov5"
DEFAULT_WEIGHTS = DEFAULT_YOLO_DIR / "yolov5s.pt"
OUTPUT_BASELINE_VERSION = "traffic-light-output-v1"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 215, 255),
    "green": (0, 180, 0),
    "unknown": (150, 150, 150),
}

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from traffic_light_classifier import TrafficLightClassifier
from traffic_light_detector import TrafficLightDetector


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def _create_run_dir(output_root, stem):
    output_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    base = f"{now.strftime('%Y%m%d_%H%M%S')}_{stem}"
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _build_tree(run_dir, stem):
    sample = run_dir / "samples" / stem
    paths = {
        "run": run_dir / "run.json",
        "sample": sample,
        "input": sample / "00_input" / "original.png",
        "detection_image": sample / "01_detection" / "detection.jpg",
        "detections": sample / "01_detection" / "detections.json",
        "raw_crops": sample / "02_classification" / "crops_raw",
        "annotated_crops": sample / "02_classification" / "crops_annotated",
        "classifications": sample / "02_classification" / "classifications.json",
        "result_image": sample / "03_result" / "result.jpg",
        "result": sample / "03_result" / "result.json",
    }
    for directory in (
        sample / "00_input", sample / "01_detection",
        paths["raw_crops"], paths["annotated_crops"],
        sample / "02_classification", sample / "03_result",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _rectangles_intersect(first, second):
    return not (
        first[2] <= second[0] or second[2] <= first[0]
        or first[3] <= second[1] or second[3] <= first[1]
    )


def _text_box_size(lines, font_scale=0.52, thickness=2, padding=5, line_gap=4):
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in lines
    ]
    width = max(size[0] for size in sizes) + padding * 2
    line_height = max(size[1] for size in sizes)
    height = line_height * len(lines) + line_gap * (len(lines) - 1) + padding * 2
    return width, height, line_height


def _choose_label_box(image_shape, target_box, label_size, occupied, blocked):
    image_height, image_width = image_shape[:2]
    label_width, label_height = label_size
    left, top, right, bottom = target_box
    margin = 8
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    candidates = [
        (center_x - label_width // 2, top - label_height - margin),
        (left, top - label_height - margin),
        (right - label_width, top - label_height - margin),
        (center_x - label_width // 2, bottom + margin),
        (left, bottom + margin),
        (right - label_width, bottom + margin),
        (right + margin, center_y - label_height // 2),
        (left - label_width - margin, center_y - label_height // 2),
    ]

    def clipped_area(box):
        x1, y1, x2, y2 = box
        visible_x1, visible_y1 = max(0, x1), max(0, y1)
        visible_x2, visible_y2 = min(image_width, x2), min(image_height, y2)
        visible = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
        total = max(1, (x2 - x1) * (y2 - y1))
        return total - visible

    scored = []
    for x, y in candidates:
        box = (x, y, x + label_width, y + label_height)
        overlap_count = sum(_rectangles_intersect(box, other) for other in occupied)
        blocked_count = sum(_rectangles_intersect(box, other) for other in blocked)
        out_of_frame = clipped_area(box)
        distance = abs(x - left) + abs(y - top)
        score = overlap_count * 100000 + blocked_count * 10000 + out_of_frame * 100 + distance
        scored.append((score, box))
    return min(scored, key=lambda item: item[0])[1]


def _draw_label(image, lines, target_box, color, occupied, blocked):
    font_scale = 0.52
    thickness = 2
    padding = 5
    line_gap = 4
    label_width, label_height, line_height = _text_box_size(
        lines, font_scale, thickness, padding, line_gap
    )
    label_box = _choose_label_box(
        image.shape, target_box, (label_width, label_height), occupied, blocked
    )
    x1, y1, x2, y2 = label_box
    image_height, image_width = image.shape[:2]
    draw_x1, draw_y1 = max(0, x1), max(0, y1)
    draw_x2, draw_y2 = min(image_width - 1, x2), min(image_height - 1, y2)
    cv2.rectangle(image, (draw_x1, draw_y1), (draw_x2, draw_y2), (25, 25, 25), -1)
    cv2.rectangle(image, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)

    text_y = y1 + padding + line_height
    for line in lines:
        cv2.putText(
            image, line, (x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA,
        )
        text_y += line_height + line_gap

    left, top, right, bottom = target_box
    target_anchor = (
        max(left, min((x1 + x2) // 2, right)),
        max(top, min((y1 + y2) // 2, bottom)),
    )
    label_anchor = (
        max(x1, min(target_anchor[0], x2)),
        max(y1, min(target_anchor[1], y2)),
    )
    cv2.line(image, target_anchor, label_anchor, color, 1, cv2.LINE_AA)
    occupied.append(label_box)


def _draw_detection(image, detection, index, color=(0, 210, 255), classified=None,
                    occupied=None, blocked=None):
    del index  # The stable tl_XXXX identifier remains in JSON, not in images.
    left, top, right, bottom = detection["bbox2d_xyxy_px"]
    target_box = (left, top, right, bottom)
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    if classified is None:
        lines = [
            "trafficlight",
            f"det={detection['confidence']:.2f}",
        ]
    else:
        lines = [
            f"trafficlight {classified['state']}",
            f"det={detection['confidence']:.2f} "
            f"cls={classified.get('model_confidence', 0.0):.2f}",
        ]
    _draw_label(
        image, lines, target_box, color,
        [] if occupied is None else occupied,
        [target_box] if blocked is None else blocked,
    )


def _annotated_crop(crop_bgr, identifier, classification):
    sidebar = 250
    canvas = np.full(
        (max(crop_bgr.shape[0], 92), crop_bgr.shape[1] + sidebar, 3),
        245, dtype=np.uint8,
    )
    canvas[:crop_bgr.shape[0], :crop_bgr.shape[1]] = crop_bgr
    state = classification.get("state", "unknown")
    color = STATE_COLORS[state]
    lines = [
        "trafficlight",
        f"state: {state}",
        f"confidence: {classification.get('model_confidence', 0.0):.3f}",
    ]
    if classification.get("reason"):
        lines.append(f"reason: {classification['reason']}")
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (crop_bgr.shape[1] + 10, 25 + line_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            color if line_index == 1 else (40, 40, 40), 2, cv2.LINE_AA,
        )
    return canvas


def run_pipeline(
    source,
    yolo_dir=DEFAULT_YOLO_DIR,
    weights=DEFAULT_WEIGHTS,
    model_dir=HERE / "models",
    imgsz=640,
    device="0",
    conf_thres=0.25,
    iou_thres=0.45,
    min_confidence=0.70,
    output_root=PROJECT_ROOT / "outputs",
):
    source = Path(source).expanduser().resolve()
    yolo_dir = Path(yolo_dir).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Source must be one readable image file: {source}")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("conf-thres, iou-thres and min-confidence must be in [0, 1]")

    run_dir = _create_run_dir(output_root / "traffic_light", source.stem)
    paths = _build_tree(run_dir, source.stem)
    started = time.perf_counter()
    stage_times = {}
    run_record = {
        "schema_version": OUTPUT_BASELINE_VERSION,
        "status": "error",
        "input": {"source": str(source), "filename": source.name, "stem": source.stem},
        "models": {
            "detector": {"framework": "YOLOv5 v6.2", "yolo_dir": str(yolo_dir), "weights": str(weights)},
            "classifier": {"framework": "OpenCV DNN", "model_dir": str(model_dir)},
        },
        "parameters": {
            "imgsz": imgsz, "device_requested": str(device),
            "conf_thres": conf_thres, "iou_thres": iou_thres,
            "min_confidence": min_confidence, "precision": "FP32",
        },
        "time": {"started_local": datetime.now().astimezone().isoformat()},
        "sample_count": 1,
        "traffic_light_count": 0,
        "error_summary": [],
        "output_baseline_version": OUTPUT_BASELINE_VERSION,
        "output_directory": str(run_dir),
        "final_result": str(paths["result"]),
    }

    try:
        input_started = time.perf_counter()
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _save_image(paths["input"], bgr)
        stage_times["input_ms"] = (time.perf_counter() - input_started) * 1000

        detection_started = time.perf_counter()
        detector = TrafficLightDetector(
            yolo_dir=yolo_dir, weights=weights, image_size=imgsz, device=device,
            confidence_threshold=conf_thres, iou_threshold=iou_thres,
        )
        detections = detector.detect(rgb)
        detections = sorted(
            detections,
            key=lambda item: (
                (item["bbox2d_xyxy_px"][0] + item["bbox2d_xyxy_px"][2]) / 2,
                (item["bbox2d_xyxy_px"][1] + item["bbox2d_xyxy_px"][3]) / 2,
            ),
        )
        for index, detection in enumerate(detections, 1):
            detection["traffic_light_id"] = f"tl_{index:04d}"
        detection_image = bgr.copy()
        detection_boxes = [tuple(item["bbox2d_xyxy_px"]) for item in detections]
        detection_labels = []
        for index, detection in enumerate(detections, 1):
            _draw_detection(
                detection_image, detection, index,
                occupied=detection_labels, blocked=detection_boxes,
            )
        _save_image(paths["detection_image"], detection_image)
        _write_json(paths["detections"], {
            "schema_version": "traffic-light-detections-0.1",
            "source": str(source),
            "image_width_px": int(rgb.shape[1]),
            "image_height_px": int(rgb.shape[0]),
            "status": "ok",
            "detections": detections,
        })
        stage_times["detection_ms"] = (time.perf_counter() - detection_started) * 1000

        classifications = []
        classification_started = time.perf_counter()
        classifier = None
        classifier_error = None
        if detections:
            try:
                classifier = TrafficLightClassifier(model_dir=model_dir, min_confidence=min_confidence)
            except Exception as error:
                classifier_error = error
                run_record["error_summary"].append({
                    "stage": "classification_init", "type": type(error).__name__,
                    "message": str(error),
                })

        for detection in detections:
            identifier = detection["traffic_light_id"]
            left, top, right, bottom = detection["bbox2d_xyxy_px"]
            crop_rgb = rgb[top:bottom, left:right]
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            raw_path = paths["raw_crops"] / f"{identifier}.png"
            annotated_path = paths["annotated_crops"] / f"{identifier}.jpg"
            _save_image(raw_path, crop_bgr)
            try:
                if classifier_error is not None:
                    raise classifier_error
                prediction = classifier.classify(crop_rgb)
                record = {
                    "traffic_light_id": identifier,
                    "status": "ok",
                    "prediction_valid": True,
                    "state": prediction["state"],
                    "model_confidence": prediction["model_confidence"],
                    "reason": prediction["reason"],
                    "min_confidence": prediction["min_confidence"],
                    "model_debug": prediction["model_debug"],
                    "crop_bbox2d_xyxy_px": [left, top, right, bottom],
                    "crop_width_px": int(right - left),
                    "crop_height_px": int(bottom - top),
                    "raw_crop_path": str(raw_path.relative_to(run_dir)),
                    "annotated_crop_path": str(annotated_path.relative_to(run_dir)),
                }
            except Exception as error:
                if classifier_error is None:
                    run_record["error_summary"].append({
                        "stage": "classification", "traffic_light_id": identifier,
                        "type": type(error).__name__, "message": str(error),
                    })
                record = {
                    "traffic_light_id": identifier, "status": "error",
                    "prediction_valid": False, "state": "unknown",
                    "model_confidence": 0.0, "reason": "classification_error",
                    "error": str(error),
                    "crop_bbox2d_xyxy_px": [left, top, right, bottom],
                    "crop_width_px": int(right - left),
                    "crop_height_px": int(bottom - top),
                    "raw_crop_path": str(raw_path.relative_to(run_dir)),
                    "annotated_crop_path": str(annotated_path.relative_to(run_dir)),
                }
            classifications.append(record)
            _save_image(annotated_path, _annotated_crop(crop_bgr, identifier, record))
        _write_json(paths["classifications"], {
            "schema_version": "traffic-light-classifications-0.1",
            "source": str(source),
            "classifications": classifications,
        })
        stage_times["classification_ms"] = (time.perf_counter() - classification_started) * 1000

        result_items = []
        for detection, classification in zip(detections, classifications):
            result_items.append({
                **detection,
                "crop_bbox2d_xyxy_px": classification["crop_bbox2d_xyxy_px"],
                "classification": classification,
            })
        result = {
            "schema_version": "traffic-light-result-0.1",
            "source": str(source),
            "image_width_px": int(rgb.shape[1]),
            "image_height_px": int(rgb.shape[0]),
            "status": "ok" if not run_record["error_summary"] else "partial_error",
            "traffic_light_count": len(result_items),
            "traffic_lights": result_items,
        }
        result_image = bgr.copy()
        result_labels = []
        for index, (detection, classification) in enumerate(zip(detections, classifications), 1):
            _draw_detection(
                result_image, detection, index, STATE_COLORS[classification["state"]],
                classification, occupied=result_labels, blocked=detection_boxes,
            )
        _save_image(paths["result_image"], result_image)
        _write_json(paths["result"], result)
        run_record["status"] = result["status"]
        run_record["traffic_light_count"] = len(result_items)
        run_record["device_actual"] = str(detector.device)
        run_record["stage_times_ms"] = stage_times
    except Exception as error:
        run_record["error_summary"].append({
            "stage": "pipeline", "type": type(error).__name__, "message": str(error),
        })
        run_record["status"] = "error"
        _write_json(paths["run"], run_record)
        raise
    finally:
        run_record["time"]["finished_local"] = datetime.now().astimezone().isoformat()
        run_record["time"]["elapsed_ms"] = (time.perf_counter() - started) * 1000
        _write_json(paths["run"], run_record)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Single-image YOLOv5 traffic-light detection and color classification")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--model-dir", type=Path, default=HERE / "models")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    try:
        run_dir = run_pipeline(**vars(args))
    except Exception as error:
        print(f"Pipeline failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
