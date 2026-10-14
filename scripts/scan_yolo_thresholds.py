"""Scan YOLO confidence thresholds on an existing CARLA episode offline."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPOSITORY_ROOT / "yolov5"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from perception_detector import YoloV5Detector  # noqa: E402
from scripts.evaluate_yolo_truth import _iou, _project_boxes  # noqa: E402


def _match(predictions, truths, threshold):
    totals = {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0}
    per_class = {}
    used = set()

    def stats(name):
        if name not in per_class:
            per_class[name] = {"truth": 0, "tp": 0, "fp": 0, "fn": 0,
                               "matched_iou_sum": 0.0}
        return per_class[name]

    for truth in truths:
        stats(truth["class"])["truth"] += 1
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        candidates = [
            (index, _iou(prediction["bbox"], truth["bbox"]))
            for index, truth in enumerate(truths)
            if index not in used and truth["class"] == prediction["class"]
        ]
        current = stats(prediction["class"])
        if candidates:
            index, score = max(candidates, key=lambda item: item[1])
            if score >= threshold:
                used.add(index)
                totals["tp"] += 1
                totals["matched_iou_sum"] += score
                current["tp"] += 1
                current["matched_iou_sum"] += score
                continue
        totals["fp"] += 1
        current["fp"] += 1
    for index, truth in enumerate(truths):
        if index not in used:
            totals["fn"] += 1
            stats(truth["class"])["fn"] += 1
    return totals, per_class


def _finish_metrics(totals, per_class, frames, visible_frames, threshold):
    for current in per_class.values():
        current["precision"] = (current["tp"] / float(current["tp"] + current["fp"])
                                 if current["tp"] + current["fp"] else 0.0)
        current["recall"] = (current["tp"] / float(current["tp"] + current["fn"])
                              if current["tp"] + current["fn"] else 0.0)
        current["f1"] = (2.0 * current["precision"] * current["recall"] /
                          (current["precision"] + current["recall"])
                          if current["precision"] + current["recall"] else 0.0)
        current["mean_matched_iou"] = (current["matched_iou_sum"] / current["tp"]
                                        if current["tp"] else 0.0)
    return {
        "confidence_threshold": threshold,
        "frames": frames,
        "frames_with_visible_truth": visible_frames,
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": (totals["tp"] / float(totals["tp"] + totals["fp"])
                      if totals["tp"] + totals["fp"] else 0.0),
        "recall": (totals["tp"] / float(totals["tp"] + totals["fn"])
                   if totals["tp"] + totals["fn"] else 0.0),
        "f1": (2.0 * totals["tp"] /
               (2.0 * totals["tp"] + totals["fp"] + totals["fn"])
               if 2 * totals["tp"] + totals["fp"] + totals["fn"] else 0.0),
        "mean_matched_iou": (totals["matched_iou_sum"] / totals["tp"]
                              if totals["tp"] else 0.0),
        "per_class": per_class,
    }


def scan_episode(episode, weights, calibration, thresholds, device="cpu", image_size=640):
    episode = Path(episode)
    with Path(calibration).open(encoding="utf-8") as stream:
        calibration_payload = json.load(stream)
    truth_records = {}
    with (episode / "truth.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            truth_records[(record["episode_id"], record["frame_id"])] = record
    image_paths = sorted((episode / "yolo_debug" / "raw").glob("frame_*.png"))
    if not image_paths:
        raise ValueError("episode has no yolo_debug/raw images")
    detector = YoloV5Detector(weights=weights, image_size=image_size, device=device,
                               confidence_threshold=min(thresholds))
    all_results = []
    for threshold in thresholds:
        totals = {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0}
        per_class = {}
        latencies = []
        visible_frames = 0
        processed = 0
        for path in image_paths:
            frame_id = int(path.stem.split("_")[-1])
            record = next((item for item in truth_records.values() if item["frame_id"] == frame_id), None)
            if record is None:
                continue
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise IOError("cannot read {}".format(path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            raw = detector.detect(image_rgb, confidence_threshold=threshold)
            latencies.append((time.perf_counter() - started) * 1000.0)
            predictions = [{"class": item["class"], "bbox": item["bbox2d_xyxy_px"],
                            "confidence": item["confidence"]} for item in raw]
            truths = _project_boxes(record["truth"], calibration_payload)
            if truths:
                visible_frames += 1
            current, current_classes = _match(predictions, truths, 0.5)
            for key in totals:
                totals[key] += current[key]
            for name, values in current_classes.items():
                target = per_class.setdefault(name, {"truth": 0, "tp": 0, "fp": 0, "fn": 0,
                                                      "matched_iou_sum": 0.0})
                for key in target:
                    target[key] += values[key]
            processed += 1
        metrics = _finish_metrics(totals, per_class, processed, visible_frames, threshold)
        metrics["latency_ms"] = {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        }
        all_results.append(metrics)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--weights", type=Path, default=YOLO_ROOT / "yolov5s.pt")
    parser.add_argument("--calibration", type=Path, default=REPOSITORY_ROOT / "calibration/front_rgb.json")
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("thresholds must be in [0, 1]")
    results = scan_episode(args.episode, args.weights, args.calibration, thresholds, args.device)
    report = {"episode": str(args.episode.resolve()), "results": results}
    output = args.episode / "yolo_threshold_scan.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
