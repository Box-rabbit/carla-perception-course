"""Validate the CARLA front_rgb pinhole projection independently of data logs.

The script creates temporary red vehicles at known positions, projects their
CARLA bounding boxes with the configured camera intrinsics and actual sensor
transform, then compares those boxes with red pixels in the captured RGB image.
It writes only a validation report and debug image under outputs/; it never
calls PerceptionRuntime or writes perception.jsonl/truth.jsonl.
"""

import argparse
import json
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import load_runtime_config, validate_python_runtime, validate_sensor_contract


def _load_calibration(path):
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("camera_id") != "front_rgb":
        raise ValueError("calibration file must describe front_rgb")
    size = payload.get("image_size_px")
    if size != [1280, 720]:
        raise ValueError("this validator currently expects [1280, 720]")
    K = np.asarray(payload.get("K"), dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("calibration K must be a finite 3x3 matrix")
    return payload, K


def _carla_transform(carla, values):
    return carla.Transform(
        carla.Location(
            x=float(values.get("x_m", 0.0)),
            y=float(values.get("y_m", 0.0)),
            z=float(values.get("z_m", 0.0)),
        ),
        carla.Rotation(
            roll=float(values.get("roll_deg", 0.0)),
            pitch=float(values.get("pitch_deg", 0.0)),
            yaw=float(values.get("yaw_deg", 0.0)),
        ),
    )


def _relative_location(ego_transform, forward_m, right_m, z_m=0.6):
    yaw = math.radians(float(ego_transform.rotation.yaw))
    return type(ego_transform.location)(
        x=float(ego_transform.location.x + math.cos(yaw) * forward_m - math.sin(yaw) * right_m),
        y=float(ego_transform.location.y + math.sin(yaw) * forward_m + math.cos(yaw) * right_m),
        z=float(ego_transform.location.z + z_m),
    )


def _world_to_camera(sensor, world_point):
    matrix = np.asarray(sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
    world = np.array([world_point.x, world_point.y, world_point.z, 1.0], dtype=np.float64)
    # CARLA/Unreal local axes are x-forward, y-right, z-up. Convert to the
    # usual pinhole axes x-right, y-down, z-forward for K projection.
    carla_camera = matrix.dot(world)
    return np.array([carla_camera[1], -carla_camera[2], carla_camera[0]], dtype=np.float64)


def _project(sensor, world_point, K, width, height):
    camera = _world_to_camera(sensor, world_point)
    if camera[2] <= 1e-6:
        return None
    uvw = K.dot(camera / camera[2])
    u, v = float(uvw[0]), float(uvw[1])
    return {"u_px": u, "v_px": v, "depth_m": float(camera[2])}


def _project_bbox(sensor, actor, K, width, height):
    vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    points = [_project(sensor, vertex, K, width, height) for vertex in vertices]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return {
        "left": max(0.0, min(point["u_px"] for point in points)),
        "top": max(0.0, min(point["v_px"] for point in points)),
        "right": min(float(width - 1), max(point["u_px"] for point in points)),
        "bottom": min(float(height - 1), max(point["v_px"] for point in points)),
        "vertices": points,
    }


def _red_bbox(rgb, region):
    """Extract the high-contrast validation target within a tight ROI."""
    red = (
        (rgb[:, :, 0] > 150)
        & (rgb[:, :, 0] > rgb[:, :, 1] * 1.35)
        & (rgb[:, :, 0] > rgb[:, :, 2] * 1.35)
    )
    left, top, right, bottom = region
    left, top = max(0, left), max(0, top)
    right, bottom = min(rgb.shape[1] - 1, right), min(rgb.shape[0] - 1, bottom)
    ys, xs = np.where(red[top:bottom + 1, left:right + 1])
    if len(xs) < 20:
        return None
    return {
        "left": int(left + xs.min()),
        "top": int(top + ys.min()),
        "right": int(left + xs.max()),
        "bottom": int(top + ys.max()),
        "pixel_count": int(len(xs)),
    }


def _center(box):
    return ((box["left"] + box["right"]) / 2.0, (box["top"] + box["bottom"]) / 2.0)


def _iou(first, second):
    left = max(first["left"], second["left"])
    top = max(first["top"], second["top"])
    right = min(first["right"], second["right"])
    bottom = min(first["bottom"], second["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first["right"] - first["left"]) * max(0.0, first["bottom"] - first["top"])
    area_second = max(0.0, second["right"] - second["left"]) * max(0.0, second["bottom"] - second["top"])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def run(args):
    config = load_runtime_config()
    system = config["_system"]
    validate_python_runtime(system)
    validate_sensor_contract(system, config["_sensors"])
    sensor_spec = config["_sensors"]["sensors"]["front_rgb"]
    calibration_path = REPOSITORY_ROOT / sensor_spec["calibration_file"]
    calibration, K = _load_calibration(calibration_path)
    width, height = [int(value) for value in sensor_spec["image_size_px"]]

    add_carla_pythonapi(args.carla_root or system["carla"].get("python_api_root"))
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)

    original_settings = world.get_settings()
    ego = None
    camera = None
    targets = []
    output_root = REPOSITORY_ROOT / "outputs" / "calibration_validation" / "front_rgb"
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(ego_bp, world.get_map().get_spawn_points()[0])
        if ego is None:
            raise RuntimeError("could not spawn ego vehicle")
        # CARLA publishes the spawn transform on the first simulation tick.
        # Read it only after that tick; before then actor transforms can still
        # contain the default origin.
        world.tick()
        ego_transform = ego.get_transform()

        target_bp = world.get_blueprint_library().find("vehicle.audi.tt")
        if target_bp.has_attribute("color"):
            target_bp.set_attribute("color", "255,0,0")
        for index, (forward_m, right_m) in enumerate(((12.0, -3.5), (20.0, 0.0), (30.0, 3.5)), 1):
            transform = carla.Transform(
                _relative_location(ego_transform, forward_m, right_m),
                carla.Rotation(yaw=float(ego_transform.rotation.yaw)),
            )
            actor = world.try_spawn_actor(target_bp, transform)
            if actor is not None:
                targets.append(("target_%02d" % index, actor, forward_m, right_m))
        if len(targets) < 3:
            raise RuntimeError("could not spawn all validation targets")
        # Publish all target transforms before using them for projection or
        # allowing the reference semantic camera to render them.
        world.tick()

        camera_bp = world.get_blueprint_library().find(sensor_spec["blueprint"])
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(float(sensor_spec["fov_deg"])))
        camera_bp.set_attribute("sensor_tick", str(system["runtime"]["sensor_tick_seconds"]))
        camera = world.spawn_actor(
            camera_bp,
            _carla_transform(carla, sensor_spec["transform"]),
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        if camera is None:
            raise RuntimeError("could not spawn front_rgb validation camera")
        captured = []
        camera.listen(lambda image: captured.append(image))
        warmup = int(system["runtime"].get("sensor_warmup_ticks", 20))
        for _ in range(warmup):
            world.tick()
        for _ in range(8):
            target_frame = int(world.tick())
            time.sleep(0.15)
        rgb_by_frame = {int(item.frame): item for item in captured}
        if target_frame not in rgb_by_frame:
            raise RuntimeError(
                f"front_rgb did not produce exact frame {target_frame}; "
                f"received={sorted(rgb_by_frame)}"
            )
        image = rgb_by_frame[target_frame]
        if int(image.width) != width or int(image.height) != height:
            raise RuntimeError("captured front_rgb image dimensions do not match config")
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((height, width, 4))
        rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
        debug_path = output_dir / "front_rgb.png"
        cv2.imwrite(str(debug_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        measurements = []
        for name, actor, forward_m, right_m in targets:
            projected = _project_bbox(camera, actor, K, width, height)
            if projected is None:
                measurements.append({"target": name, "status": "behind_camera"})
                continue
            # Keep the ROI tight so nearby validation vehicles cannot be
            # mistaken for this target.
            margin = 8
            region = (
                max(0, int(projected["left"] - margin)),
                max(0, int(projected["top"] - margin)),
                min(width - 1, int(projected["right"] + margin)),
                min(height - 1, int(projected["bottom"] + margin)),
            )
            observed = _red_bbox(rgb, region)
            record = {
                "target": name,
                "placement_m": {"forward": forward_m, "right": right_m},
                "projected_bbox_px": {key: value for key, value in projected.items() if key != "vertices"},
                "observed_red_bbox_px": observed,
            }
            if observed is None:
                record.update({"status": "no_red_pixels", "center_error_px": None, "bbox_iou": 0.0})
            else:
                expected_center = _center(projected)
                observed_center = _center(observed)
                error = math.hypot(expected_center[0] - observed_center[0], expected_center[1] - observed_center[1])
                record.update({
                    "status": "ok" if error <= args.max_center_error_px else "high_pixel_error",
                    "center_error_px": error,
                    "bbox_iou": _iou(projected, observed),
                })
            measurements.append(record)

        successful = [item for item in measurements if item.get("center_error_px") is not None]
        max_error = max((item["center_error_px"] for item in successful), default=None)
        report = {
            "schema_version": "front-rgb-projection-validation-0.1",
            "camera_id": "front_rgb",
            "map": world.get_map().name,
            "captured_frame": int(image.frame),
            "target_frame_requested": target_frame,
            "reference": "tight red validation target pixels in the RGB image",
            "image_size_px": [int(image.width), int(image.height)],
            "calibration_file": str(calibration_path),
            "K": K.tolist(),
            "sensor_transform": {
                "location": {"x": camera.get_transform().location.x, "y": camera.get_transform().location.y, "z": camera.get_transform().location.z},
                "rotation": {"roll": camera.get_transform().rotation.roll, "pitch": camera.get_transform().rotation.pitch, "yaw": camera.get_transform().rotation.yaw},
            },
            "max_center_error_px": max_error,
            "pass": bool(successful) and max_error <= args.max_center_error_px,
            "thresholds": {"max_center_error_px": args.max_center_error_px},
            "measurements": measurements,
            "debug_image": str(debug_path),
            "truth_log_written": False,
            "perception_log_written": False,
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["pass"] else 2
    finally:
        if camera is not None and camera.is_alive:
            camera.stop()
            camera.destroy()
        for _name, actor, _forward, _right in targets:
            if actor.is_alive:
                actor.destroy()
        if ego is not None and ego.is_alive:
            ego.destroy()
        world.apply_settings(original_settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--carla-root")
    parser.add_argument("--max-center-error-px", type=float, default=35.0)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()
