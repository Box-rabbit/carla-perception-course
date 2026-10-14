"""Evaluate online YOLO boxes against the separate CARLA truth log.

The evaluator is offline only. It never supplies truth to the detector.
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_carla():
    from common.carla import add_carla_pythonapi
    add_carla_pythonapi()
    import carla
    return carla


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _project_boxes(truth, calibration):
    carla = _load_carla()
    camera = truth.get("camera")
    if not camera:
        return []
    t = camera["transform"]
    transform = carla.Transform(
        carla.Location(**t["location"]),
        carla.Rotation(**t["rotation"]),
    )
    inverse = np.asarray(transform.get_inverse_matrix(), dtype=np.float64)
    K = np.asarray(calibration["K"], dtype=np.float64)
    width, height = calibration["image_size_px"]
    boxes = []
    ego_actor_id = truth.get("ego", {}).get("actor_id")
    for actor in truth.get("actors", []):
        if actor.get("actor_id") == ego_actor_id:
            continue
        vertices = actor.get("bbox_world_vertices")
        if not vertices or not actor.get("type_id", "").startswith(("vehicle.", "walker.")):
            continue
        points = []
        for item in vertices:
            world = np.array([item["x"], item["y"], item["z"], 1.0])
            local = inverse.dot(world)
            camera_point = np.array([local[1], -local[2], local[0]])
            if camera_point[2] <= 1e-6:
                continue
            uv = K.dot(camera_point / camera_point[2])
            points.append((float(uv[0]), float(uv[1])))
        if not points:
            continue
        box = [max(0.0, min(p[0] for p in points)), max(0.0, min(p[1] for p in points)),
               min(float(width - 1), max(p[0] for p in points)),
               min(float(height - 1), max(p[1] for p in points))]
        if box[2] > box[0] and box[3] > box[1]:
            category = "pedestrian" if actor["type_id"].startswith("walker.") else "vehicle"
            boxes.append({"class": category, "bbox": box})
    return boxes


def evaluate_episode(episode, calibration_path, iou_threshold=0.5):
    episode = Path(episode)
    with Path(calibration_path).open(encoding="utf-8") as stream:
        calibration = json.load(stream)
    perception_path, truth_path = episode / "perception.jsonl", episode / "truth.jsonl"
    perceptions = {}
    with perception_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            perceptions[(item["episode_id"], item["frame_id"])] = item
    totals = {
        "frames": 0,
        "frames_with_visible_truth": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "matched_iou_sum": 0.0,
    }
    per_class = {}

    def class_stats(name):
        if name not in per_class:
            per_class[name] = {
                "truth": 0, "tp": 0, "fp": 0, "fn": 0,
                "matched_iou_sum": 0.0,
            }
        return per_class[name]

    with truth_path.open(encoding="utf-8") as stream:
        for line in stream:
            truth_record = json.loads(line)
            key = (truth_record["episode_id"], truth_record["frame_id"])
            perception = perceptions.get(key)
            if perception is None:
                continue
            yolo = perception.get("results", {}).get("yolov5", {}).get("result") or {}
            predictions = [
                {"class": d["class"], "bbox": d["bbox2d_xyxy_px"],
                 "confidence": d.get("confidence", 0.0)}
                for d in yolo.get("detections", [])
            ]
            truths = _project_boxes(truth_record["truth"], calibration)
            if truths:
                totals["frames_with_visible_truth"] += 1
            for truth in truths:
                class_stats(truth["class"])["truth"] += 1
            used = set()
            for pred in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
                candidates = [
                    (index, _iou(pred["bbox"], gt["bbox"]))
                    for index, gt in enumerate(truths)
                    if index not in used and gt["class"] == pred["class"]
                ]
                stats = class_stats(pred["class"])
                if candidates:
                    index, score = max(candidates, key=lambda x: x[1])
                    if score >= iou_threshold:
                        used.add(index)
                        totals["tp"] += 1
                        totals["matched_iou_sum"] += score
                        stats["tp"] += 1
                        stats["matched_iou_sum"] += score
                        continue
                totals["fp"] += 1
                stats["fp"] += 1
            for index, truth in enumerate(truths):
                if index not in used:
                    totals["fn"] += 1
                    class_stats(truth["class"])["fn"] += 1
            totals["frames"] += 1

    for stats in per_class.values():
        stats["precision"] = (
            stats["tp"] / float(stats["tp"] + stats["fp"])
            if stats["tp"] + stats["fp"] else 0.0
        )
        stats["recall"] = (
            stats["tp"] / float(stats["tp"] + stats["fn"])
            if stats["tp"] + stats["fn"] else 0.0
        )
        stats["f1"] = (
            2.0 * stats["precision"] * stats["recall"] /
            (stats["precision"] + stats["recall"])
            if stats["precision"] + stats["recall"] else 0.0
        )
        stats["mean_matched_iou"] = (
            stats["matched_iou_sum"] / stats["tp"] if stats["tp"] else 0.0
        )
    report = dict(
        totals,
        iou_threshold=iou_threshold,
        evaluation_status=(
            "ok" if totals["frames_with_visible_truth"] else "no_visible_truth"
        ),
        precision=(totals["tp"] / float(totals["tp"] + totals["fp"])
                   if totals["tp"] + totals["fp"] else 0.0),
        recall=(totals["tp"] / float(totals["tp"] + totals["fn"])
                if totals["tp"] + totals["fn"] else 0.0),
        mean_matched_iou=(
            totals["matched_iou_sum"] / totals["tp"] if totals["tp"] else 0.0
        ),
        per_class=per_class,
    )
    report["f1"] = (
        2.0 * report["precision"] * report["recall"] /
        (report["precision"] + report["recall"])
        if report["precision"] + report["recall"] else 0.0
    )
    with (episode / "yolo_truth_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--calibration", type=Path, default=REPOSITORY_ROOT / "calibration/front_rgb.json")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    report = evaluate_episode(args.episode, args.calibration, iou_threshold=args.iou)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
