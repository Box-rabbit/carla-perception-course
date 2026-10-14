"""Run the smallest end-to-end CARLA sensor/perception data chain.

The runner owns transport only: CARLA sensors produce ``SensorFrameBundle``
objects, the perception runtime consumes those bundles, and scenario truth is
written through a separate logger call.  No truth object is passed to a model
processor.
"""

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import statistics
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import (
    DEFAULT_SCENARIO_CONFIG,
    enabled_sensors,
    load_runtime_config,
    validate_python_runtime,
    validate_sensor_contract,
)
from .logger import EpisodeLogger
from .runtime import PerceptionRuntime
from .sensor_bridge import CarlaSensorBridge
from .yolo_processor import build_yolo_processor


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


def _spawn_ego(carla, world, config):
    ego_cfg = config.get("ego", {})
    spawn = ego_cfg.get("spawn_point", {})
    transform = carla.Transform(
        carla.Location(
            x=float(spawn.get("x", 0.0)),
            y=float(spawn.get("y", 0.0)),
            z=float(spawn.get("z", 0.5)),
        ),
        carla.Rotation(yaw=float(spawn.get("yaw", 0.0))),
    )
    blueprint = world.get_blueprint_library().find(
        ego_cfg.get("vehicle_type", "vehicle.tesla.model3")
    )
    ego = world.try_spawn_actor(blueprint, transform)
    if ego is None:
        raise RuntimeError(f"failed to spawn ego at {transform.location}")
    return ego


def _configure_camera(carla, world, ego, camera_id, spec, sensor_tick):
    blueprint = world.get_blueprint_library().find(spec["blueprint"])
    width, height = [int(value) for value in spec["image_size_px"]]
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(float(spec["fov_deg"])))
    blueprint.set_attribute("sensor_tick", str(float(sensor_tick)))
    sensor = world.spawn_actor(
        blueprint,
        _carla_transform(carla, spec["transform"]),
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )
    if sensor is None:
        raise RuntimeError(f"failed to spawn sensor {camera_id}")
    return sensor


def _spawn_evaluation_targets(carla, world, ego, profile="vehicle"):
    """Spawn deterministic visible actors for the optional YOLO evaluation run."""
    transform = ego.get_transform()
    yaw = math.radians(float(transform.rotation.yaw))
    specs = [("vehicle.tesla.model3", 18.0, 0.0, 0.5, "vehicle")]
    if profile == "mixed":
        specs.extend([
            ("vehicle.audi.tt", 30.0, -2.5, 0.5, "vehicle"),
            ("walker.pedestrian.0001", 20.0, 2.5, 1.0, "pedestrian"),
        ])
    targets = []
    for blueprint_id, distance, lateral, height, _category in specs:
        location = carla.Location(
            x=float(transform.location.x + math.cos(yaw) * distance - math.sin(yaw) * lateral),
            y=float(transform.location.y + math.sin(yaw) * distance + math.cos(yaw) * lateral),
            z=float(transform.location.z + height),
        )
        target = world.try_spawn_actor(
            world.get_blueprint_library().find(blueprint_id),
            carla.Transform(location, carla.Rotation(yaw=float(transform.rotation.yaw))),
        )
        if target is None:
            raise RuntimeError("failed to spawn evaluation target: {}".format(blueprint_id))
        target.set_simulate_physics(False)
        targets.append(target)
    return targets


def _location(location):
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def _rotation(rotation):
    return {
        "roll": float(rotation.roll),
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
    }


def _actor_truth(actor):
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    record = {
        "actor_id": int(actor.id),
        "type_id": actor.type_id,
        "transform": {
            "location": _location(transform.location),
            "rotation": _rotation(transform.rotation),
        },
        "velocity_mps": _location(velocity),
    }
    bbox = getattr(actor, "bounding_box", None)
    if bbox is not None:
        record["bbox_world_vertices"] = [
            _location(vertex) for vertex in bbox.get_world_vertices(transform)
        ]
    return record


def _truth_payload(world, ego, scenario_state, camera_sensor=None):
    snapshot = world.get_snapshot()
    actors = []
    for actor in world.get_actors():
        if actor.id == ego.id or actor.type_id.startswith(("vehicle.", "walker.")):
            if actor.is_alive:
                actors.append(_actor_truth(actor))
    payload = {
        "snapshot": {
            "frame_id": int(snapshot.frame),
            "timestamp_seconds": float(snapshot.timestamp.elapsed_seconds),
        },
        "ego": _actor_truth(ego),
        "actors": actors,
        "scenario_observation": scenario_state.get("scenario_observation", {}),
    }
    if camera_sensor is not None:
        camera_transform = camera_sensor.get_transform()
        payload["camera"] = {
            "camera_id": "front_rgb",
            "transform": {
                "location": _location(camera_transform.location),
                "rotation": _rotation(camera_transform.rotation),
            },
        }
    return payload


def _make_output_dir(config, episode_id):
    root = REPOSITORY_ROOT / config["_system"]["paths"]["output_root"]
    path = root / "data_chain" / episode_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(args):
    config = load_runtime_config(scenario_path=args.config)
    system = config["_system"]
    if args.remote_inference_port is not None:
        if not 1 <= args.remote_inference_port <= 65535:
            raise ValueError("remote_inference_port must be in [1, 65535]")
        config["_perception"]["inference_backend"] = "remote"
        config["_perception"]["remote_inference"] = {
            "host": args.remote_inference_host,
            "port": args.remote_inference_port,
        }
    if args.frequency_hz is not None:
        if args.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        # Frequency is a run-time override for the baseline tool. The
        # canonical 20 Hz contract remains in config/system.yaml.
        tick = 1.0 / float(args.frequency_hz)
        system["runtime"]["fixed_delta_seconds"] = tick
        system["runtime"]["sensor_tick_seconds"] = tick
    validate_python_runtime(system)
    sensors_config = config["_sensors"]
    validate_sensor_contract(system, sensors_config)
    enabled = enabled_sensors(sensors_config)
    if not enabled:
        raise RuntimeError("no enabled sensors in config/sensors.yaml")
    if args.distance_m <= 0 or args.max_frames <= 0:
        raise ValueError("distance_m and max_frames must be positive")
    if not 0.0 <= args.throttle <= 1.0:
        raise ValueError("throttle must be in [0, 1]")

    add_carla_pythonapi(args.carla_root or system.get("carla", {}).get("python_api_root"))
    try:
        import carla
        if not hasattr(carla, "Client"):
            raise ImportError("CARLA PythonAPI has no Client for this Python interpreter")
    except ImportError as error:
        raise RuntimeError(
            "CARLA PythonAPI is unavailable for this Python interpreter. "
            "Use a Python version matching CARLA's carla*.egg/carla*.whl."
        ) from error
    from My_scenario.scenarios_impl.my_scenario import build_scenario

    host = args.host or system["carla"]["host"]
    port = int(args.port or system["carla"]["port"])
    runtime = system["runtime"]
    episode_id = args.episode_id or (
        f"{runtime['episode_id_prefix']}_"
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir = _make_output_dir(config, episode_id)

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()
    requested_town = config.get("map", {}).get("town")
    if requested_town and not world.get_map().name.endswith(str(requested_town)):
        world = client.load_world(str(requested_town))
    original_settings = world.get_settings()
    ego = None
    sensors = []
    scenario = None
    evaluation_targets = []
    bridge = None
    frame_count = 0
    inference_latencies_ms = []
    total_latencies_ms = []
    tick_latencies_ms = []
    sensor_wait_latencies_ms = []
    truth_log_latencies_ms = []
    perception_runtime = None
    initial_location = None
    scenario_state = {}
    try:
        settings = world.get_settings()
        settings.synchronous_mode = bool(runtime["synchronous_mode"])
        settings.fixed_delta_seconds = float(runtime["fixed_delta_seconds"])
        world.apply_settings(settings)

        ego = _spawn_ego(carla, world, config)
        world.tick()
        initial_location = ego.get_location()
        scenario = build_scenario()
        scenario_actors = scenario.spawn_actors(world, ego, config)
        evaluation_profile = args.evaluation_profile
        if args.evaluation_targets and evaluation_profile == "none":
            evaluation_profile = "vehicle"
        if evaluation_profile != "none":
            evaluation_targets = _spawn_evaluation_targets(carla, world, ego, evaluation_profile)
            print("evaluation targets spawned: {} ({})".format(
                len(evaluation_targets), evaluation_profile
            ))

        expected_sizes = {}
        for camera_id, spec in enabled.items():
            expected_sizes[camera_id] = tuple(int(v) for v in spec["image_size_px"])
            sensors.append(
                (camera_id, _configure_camera(
                    carla, world, ego, camera_id, spec,
                    runtime["sensor_tick_seconds"],
                ))
            )
        bridge = CarlaSensorBridge(
            episode_id,
            expected_sizes=expected_sizes,
            camera_ids=list(enabled),
            queue_size=int(runtime["sensor_queue_size"]),
        )
        for camera_id, sensor in sensors:
            bridge.attach_sensor(camera_id, sensor)

        # Let CARLA finish initializing camera rendering before recording the
        # episode. Warmup frames are deliberately discarded; the first logged
        # bundle still has to match its exact world.tick() frame ID.
        warmup_ticks = int(runtime.get("sensor_warmup_ticks", 0))
        if warmup_ticks < 0:
            raise ValueError("sensor_warmup_ticks must be nonnegative")
        warmup_frame_id = None
        for _ in range(warmup_ticks):
            warmup_frame_id = int(world.tick())
        if warmup_frame_id is not None:
            bridge.get_bundle(
                warmup_frame_id,
                timeout_s=float(runtime["sensor_timeout_seconds"]),
            )
            print(
                "sensor warmup complete: "
                f"ticks={warmup_ticks}, frame_id={warmup_frame_id}, "
                f"stats={bridge.stats()}"
            )
        bridge_stats_start = bridge.stats()

        debug_dir = None
        perception_config = config.get("_perception", {})
        yolo_config = perception_config.get("yolov5", {})
        if bool(yolo_config.get("enabled", False)):
            if args.save_yolo_debug:
                debug_dir = output_dir / "yolo_debug"
            processor = build_yolo_processor(perception_config, debug_output_dir=debug_dir)
            perception_runtime = PerceptionRuntime({"yolov5": processor})
        else:
            perception_runtime = PerceptionRuntime()
        episode_wall_start = time.perf_counter()
        with EpisodeLogger(
            output_dir,
            perception_name=system["paths"]["perception_log_name"],
            truth_name=system["paths"]["truth_log_name"],
        ) as logger:
            max_sim_seconds = float(runtime["max_duration_seconds"])
            while (
                frame_count < args.max_frames
                and frame_count * float(runtime["fixed_delta_seconds"]) < max_sim_seconds
            ):
                tick_started = time.perf_counter()
                frame_id = int(world.tick())
                tick_latencies_ms.append((time.perf_counter() - tick_started) * 1000.0)
                sensor_started = time.perf_counter()
                bundle = bridge.get_bundle(
                    frame_id, timeout_s=float(runtime["sensor_timeout_seconds"])
                )
                sensor_wait_latencies_ms.append((time.perf_counter() - sensor_started) * 1000.0)
                perception = perception_runtime.process(bundle)
                logger.log_perception(perception)
                yolo_result = perception.get("results", {}).get("yolov5", {}).get("result")
                total_latency = perception.get("timing", {}).get("total_latency_ms")
                if isinstance(total_latency, (int, float)) and math.isfinite(total_latency):
                    total_latencies_ms.append(float(total_latency))
                if isinstance(yolo_result, dict):
                    latency = yolo_result.get("inference_latency_ms")
                    if isinstance(latency, (int, float)) and math.isfinite(latency):
                        inference_latencies_ms.append(float(latency))

                current = ego.get_location()
                distance = math.sqrt(
                    (current.x - initial_location.x) ** 2
                    + (current.y - initial_location.y) ** 2
                )
                obs = {
                    "speed_kmh": 0.0,
                    "route_metrics": {
                        "route_progress_m": distance,
                        "max_route_progress_m": distance,
                    },
                }
                scenario_state = scenario.update_state(
                    ego, scenario_actors, scenario_state, obs,
                    float(runtime["fixed_delta_seconds"]), config,
                )
                truth_started = time.perf_counter()
                logger.log_truth(
                    episode_id, bundle.frame_id, bundle.timestamp_ns,
                    _truth_payload(
                        world, ego, scenario_state,
                        camera_sensor=dict(sensors).get("front_rgb"),
                    ),
                )
                truth_log_latencies_ms.append((time.perf_counter() - truth_started) * 1000.0)

                # Temporary straight-line motion until planning/control is added.
                ego.apply_control(carla.VehicleControl(throttle=args.throttle, steer=0.0))
                frame_count += 1
                if distance >= args.distance_m:
                    break
        episode_wall_seconds = time.perf_counter() - episode_wall_start
        if inference_latencies_ms:
            ordered = sorted(inference_latencies_ms)
            p95_index = max(0, int(math.ceil(0.95 * len(ordered))) - 1)
            print(
                "YOLO inference latency: "
                "frames={}, mean={:.2f} ms, p95={:.2f} ms, max={:.2f} ms".format(
                    len(ordered), statistics.mean(ordered), ordered[p95_index], ordered[-1]
                )
            )
        summary = {
            "episode_id": episode_id,
            "config": str(Path(args.config).resolve()),
            "frequency_hz": 1.0 / float(runtime["fixed_delta_seconds"]),
            "fixed_delta_seconds": float(runtime["fixed_delta_seconds"]),
            "sensor_tick_seconds": float(runtime["sensor_tick_seconds"]),
            "frames_processed": frame_count,
            "episode_wall_seconds": episode_wall_seconds,
            "achieved_wall_fps": frame_count / episode_wall_seconds if episode_wall_seconds else None,
            "simulation_seconds": frame_count * float(runtime["fixed_delta_seconds"]),
            "real_time_factor": (frame_count * float(runtime["fixed_delta_seconds"]) /
                                 episode_wall_seconds) if episode_wall_seconds else None,
            "sensor_bridge": bridge.stats() if bridge is not None else {},
            "sensor_bridge_start": bridge_stats_start if bridge is not None else {},
            "inference_latency_ms": {
                "mean": statistics.mean(inference_latencies_ms) if inference_latencies_ms else None,
                "p95": (sorted(inference_latencies_ms)[max(0, int(math.ceil(0.95 * len(inference_latencies_ms))) - 1)]
                        if inference_latencies_ms else None),
                "max": max(inference_latencies_ms) if inference_latencies_ms else None,
            },
            "total_latency_ms": {
                "mean": statistics.mean(total_latencies_ms) if total_latencies_ms else None,
                "p95": (sorted(total_latencies_ms)[max(0, int(math.ceil(0.95 * len(total_latencies_ms))) - 1)]
                        if total_latencies_ms else None),
                "max": max(total_latencies_ms) if total_latencies_ms else None,
            },
            "pipeline_timing_ms": {
                "world_tick_mean": statistics.mean(tick_latencies_ms) if tick_latencies_ms else None,
                "sensor_wait_mean": statistics.mean(sensor_wait_latencies_ms) if sensor_wait_latencies_ms else None,
                "truth_log_mean": statistics.mean(truth_log_latencies_ms) if truth_log_latencies_ms else None,
            },
        }
        import json
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        return output_dir
    finally:
        if bridge is not None:
            bridge.close()
        if perception_runtime is not None:
            perception_runtime.close()
        for _camera_id, sensor in sensors:
            # CarlaSensorBridge.close() already stops listeners. Destroy the
            # actor here without calling stop() a second time; CARLA warns
            # when unsubscribe is repeated after the listener is removed.
            if sensor.is_alive:
                sensor.destroy()
        if scenario is not None:
            for actor in getattr(scenario, "_actors", []):
                if actor is not None and actor.is_alive:
                    actor.destroy()
        for actor in evaluation_targets:
            if actor is not None and actor.is_alive:
                actor.destroy()
        if ego is not None and ego.is_alive:
            ego.destroy()
        world.apply_settings(original_settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--carla-root")
    parser.add_argument("--episode-id")
    parser.add_argument("--distance-m", type=float, default=100.0)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--frequency-hz", type=float, default=None,
                        help="temporary fixed-step/sensor frequency override for baseline tests")
    parser.add_argument("--remote-inference-host", default="127.0.0.1")
    parser.add_argument("--remote-inference-port", type=int,
                        help="use separate YOLO inference server on this port")
    parser.add_argument("--throttle", type=float, default=0.35)
    parser.add_argument(
        "--save-yolo-debug", action="store_true",
        help="save raw and YOLO-annotated front_rgb PNGs for manual inspection",
    )
    parser.add_argument(
        "--evaluation-targets", action="store_true",
        help="spawn a deterministic vehicle in front of the ego for offline YOLO evaluation",
    )
    parser.add_argument(
        "--evaluation-profile", choices=("none", "vehicle", "mixed"), default="none",
        help="temporary perception fixture: one vehicle or mixed vehicles and pedestrian",
    )
    args = parser.parse_args()
    output_dir = run(args)
    print(f"data chain complete: {output_dir}")


if __name__ == "__main__":
    main()
