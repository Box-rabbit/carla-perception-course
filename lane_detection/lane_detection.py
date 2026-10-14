import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2

try:
    from ufld_detector import DEFAULT_UFLD_DIR, DEFAULT_WEIGHTS, UFLDDetector
except ImportError:
    from lane_detection.ufld_detector import DEFAULT_UFLD_DIR, DEFAULT_WEIGHTS, UFLDDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANE_COLORS = ((255, 100, 0), (0, 200, 0), (0, 180, 255), (220, 0, 220))


def draw_lanes(bgr, lanes, connect):
    canvas = bgr.copy()
    for lane in lanes:
        if not lane["valid"]:
            continue
        color = LANE_COLORS[lane["model_slot"]]
        previous = None
        for point in lane["points"]:
            coordinate = tuple(point["xy_px"])
            if connect and previous is not None:
                previous_index = previous["anchor_index_bottom_to_top"]
                if point["anchor_index_bottom_to_top"] == previous_index + 1:
                    cv2.line(canvas, tuple(previous["xy_px"]), coordinate, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, coordinate, 3, color, -1, cv2.LINE_AA)
            previous = point
    return canvas


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Image save failed: {path}")


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def make_run_dir(output_root, stem):
    module_dir = output_root / "lane_detection"
    module_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_" + stem
    suffix = 0
    while True:
        candidate = module_dir / (name if suffix == 0 else f"{name}_{suffix:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def _metadata_from_arguments(arguments):
    values = {
        "episode_id": arguments.episode_id,
        "camera_id": arguments.camera_id,
        "frame_id": arguments.frame_id,
        "timestamp_ns": arguments.timestamp_ns,
        "frame_id_source": arguments.frame_id_source,
        "timestamp_source": arguments.timestamp_source,
    }
    return values if any(value is not None for value in values.values()) else None


def _run_record(arguments, run_dir):
    return {
        "schema_version": "ufld-single-image-run-0.2",
        "status": "running",
        "started_local": datetime.now().astimezone().isoformat(),
        "source": str(arguments.source.resolve()),
        "model": "UFLD_v1_CULane_ResNet18",
        "ufld_dir": str(arguments.ufld_dir.resolve()),
        "weights": str(arguments.weights.resolve()),
        "device_requested": arguments.device,
        "trusted_legacy_checkpoint": arguments.trusted_legacy_checkpoint,
        "network_input_hw": [288, 800],
        "precision": "FP32",
        "frame_metadata": _metadata_from_arguments(arguments),
        "error_summary": [],
        "stage_times_ms": {},
        "output_directory": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="UFLD v1 CULane ResNet18 single-image inference")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ufld-dir", type=Path, default=DEFAULT_UFLD_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--trusted-legacy-checkpoint", action="store_true")
    parser.add_argument("--episode-id")
    parser.add_argument("--camera-id")
    parser.add_argument("--frame-id", type=int)
    parser.add_argument("--timestamp-ns", type=int)
    parser.add_argument("--frame-id-source")
    parser.add_argument("--timestamp-source")
    arguments = parser.parse_args()

    source = arguments.source.expanduser().resolve()
    arguments.source = source
    if not source.is_file():
        parser.error(f"Source does not exist: {source}")
    run_dir = make_run_dir(arguments.output_root.expanduser().resolve(), source.stem)
    run = _run_record(arguments, run_dir)
    save_json(run_dir / "run.json", run)
    started = time.perf_counter()
    sample = run_dir / "samples" / source.stem
    for directory in ("00_input", "01_detection", "02_result"):
        (sample / directory).mkdir(parents=True, exist_ok=True)
    image_width = None
    image_height = None

    try:
        detector = UFLDDetector(
            ufld_dir=arguments.ufld_dir,
            weights=arguments.weights,
            device=arguments.device,
            trusted_legacy_checkpoint=arguments.trusted_legacy_checkpoint,
        )
        run["device_actual"] = str(detector.device)
        run["weights_sha256"] = detector.weights_sha256
        run["reference_commit"] = detector.reference_commit
        try:
            revision = subprocess.run(
                ["git", "-C", str(detector.ufld_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            run["ufld_commit_actual"] = revision.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            run["ufld_commit_actual"] = None
        run["ignored_training_aux_parameter_count"] = detector.ignored_aux_parameter_count
        run["stage_times_ms"]["model_load_ms"] = detector.model_load_time_ms
        run["stage_times_ms"]["warmup_ms"] = detector.warmup_time_ms
        run["stage_times_ms"]["model_initialization_ms"] = detector.initialization_time_ms

        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Input image cannot be decoded: {source}")
        image_height, image_width = bgr.shape[:2]

        visualization_started = time.perf_counter()
        save_image(sample / "00_input" / "original.png", bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = detector.detect(rgb, metadata=_metadata_from_arguments(arguments))
        result["source"] = str(source)
        run["stage_times_ms"].update(result["timings_ms"])
        save_image(sample / "01_detection" / "lane_points.jpg", draw_lanes(bgr, result["lanes"], False))
        save_image(sample / "02_result" / "result.jpg", draw_lanes(bgr, result["lanes"], True))
        save_json(sample / "02_result" / "result.json", result)
        run["stage_times_ms"]["visualization_and_save_ms"] = (
            time.perf_counter() - visualization_started
        ) * 1000
        run["valid_lane_count"] = result["valid_lane_count"]
        run["lanes_valid"] = result["lanes_valid"]
        run["final_image"] = (sample / "02_result" / "result.jpg").relative_to(run_dir).as_posix()
        run["final_result"] = (sample / "02_result" / "result.json").relative_to(run_dir).as_posix()
        run["status"] = result["status"]
    except Exception as error:
        run["status"] = "error"
        run["lanes_valid"] = False
        run["error_summary"].append({"type": type(error).__name__, "message": str(error)})
        error_result = {
            "schema_version": "lane2d-0.2",
            "status": "error",
            "lanes_valid": False,
            "source": str(source),
            "metadata": _metadata_from_arguments(arguments),
            "image_width_px": image_width,
            "image_height_px": image_height,
            "coordinate_frame": "image_pixels_x_right_y_down",
            "valid_lane_count": 0,
            "lanes": [],
            "errors": [{"type": type(error).__name__, "message": str(error)}],
            "ego_boundaries_valid": False,
            "left_boundary_lane_id": None,
            "right_boundary_lane_id": None,
            "ego_boundary_role": "unknown",
            "metric_distances_valid": False,
            "left_distance_m": None,
            "right_distance_m": None,
        }
        save_json(sample / "02_result" / "result.json", error_result)
        raise
    finally:
        run["finished_local"] = datetime.now().astimezone().isoformat()
        run["elapsed_ms"] = (time.perf_counter() - started) * 1000
        save_json(run_dir / "run.json", run)
        print(f"Run directory: {run_dir}")

    print(f"Valid lane candidates: {run['valid_lane_count']}")
    print("Ego-lane boundaries and metric lateral distances are not estimated in this version.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Lane pipeline failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
