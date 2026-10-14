"""Automated acceptance checks for one online YOLO episode."""

import argparse
import json
from pathlib import Path
import statistics
import math

from scripts.evaluate_yolo_truth import evaluate_episode


FORBIDDEN = ("truth", "ground_truth", "actor_id", "scenario_observation")


def validate_episode(episode, calibration, max_latency_ms=50.0, require_targets=True,
                     require_debug_images=True):
    episode = Path(episode)
    perception_lines = (episode / "perception.jsonl").read_text(encoding="utf-8").splitlines()
    truth_lines = (episode / "truth.jsonl").read_text(encoding="utf-8").splitlines()
    if not perception_lines or not truth_lines:
        raise ValueError("perception.jsonl and truth.jsonl must both be nonempty")
    perceptions = [json.loads(line) for line in perception_lines]
    truths = [json.loads(line) for line in truth_lines]
    perception_keys = [(item["episode_id"], item["frame_id"]) for item in perceptions]
    truth_keys = [(item["episode_id"], item["frame_id"]) for item in truths]
    if perception_keys != truth_keys:
        raise AssertionError("perception/truth frame keys are not aligned")
    frame_ids = [item["frame_id"] for item in perceptions]
    if any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise AssertionError("perception frame IDs are not strictly increasing")
    if any(any(field in json.dumps(item) for field in FORBIDDEN) for item in perceptions):
        raise AssertionError("perception record contains forbidden truth field")
    statuses = [item["results"].get("yolov5", {}).get("status") for item in perceptions]
    if any(status != "ok" for status in statuses):
        raise AssertionError("YOLO processor did not remain ok for every frame")
    latencies = [float(item["results"]["yolov5"]["result"]["inference_latency_ms"])
                 for item in perceptions]
    if any(value < 0 for value in latencies):
        raise AssertionError("negative inference latency")
    report = evaluate_episode(episode, calibration)
    if require_targets and report["evaluation_status"] != "ok":
        raise AssertionError("episode has no visible truth targets")
    result = {
        "status": "pass",
        "frames": len(perceptions),
        "json_interface": "pass",
        "truth_isolation": "pass",
        "frame_alignment": "pass",
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": sorted(latencies)[max(0, int(math.ceil(0.50 * len(latencies))) - 1)],
            "p95": sorted(latencies)[max(0, int(math.ceil(0.95 * len(latencies))) - 1)],
            "max": max(latencies),
            "over_budget_frames": sum(value > max_latency_ms for value in latencies),
            "budget_ms": max_latency_ms,
        },
        "device": {
            "requested": sorted(set(
                str(item["results"]["yolov5"]["result"].get("device_requested", "unknown"))
                for item in perceptions
            )),
            "actual": sorted(set(
                str(item["results"]["yolov5"]["result"].get("device_actual", "unknown"))
                for item in perceptions
            )),
            "precision": sorted(set(
                str(item["results"]["yolov5"]["result"].get("precision", "unknown"))
                for item in perceptions
            )),
        },
        "metrics": report,
        "debug_images": {
            "raw": len(list((episode / "yolo_debug" / "raw").glob("*.png"))),
            "annotated": len(list((episode / "yolo_debug" / "annotated").glob("*.png"))),
        },
    }
    if require_debug_images and (result["debug_images"]["raw"] != len(perceptions) or result["debug_images"]["annotated"] != len(perceptions)):
        raise AssertionError("debug image count does not match frame count")
    if result["latency_ms"]["over_budget_frames"]:
        result["status"] = "pass_with_latency_over_budget"
    with (episode / "yolo_acceptance.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).resolve().parents[1] / "calibration/front_rgb.json")
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    parser.add_argument("--allow-no-targets", action="store_true")
    parser.add_argument("--allow-no-debug", action="store_true")
    args = parser.parse_args()
    result = validate_episode(args.episode, args.calibration, args.max_latency_ms,
                              require_targets=not args.allow_no_targets,
                              require_debug_images=not args.allow_no_debug)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
