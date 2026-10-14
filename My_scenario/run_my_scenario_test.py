"""旧文件：自带简单控制器 + 路线跟随，测试场景搭建是否成功。

用法：
  # 自由视角（默认）
  $PROJECT_PYTHON run_my_scenario_test.py --config configs/my_scenario.yaml --duration 120

  # 观察者跟随 ego
  $PROJECT_PYTHON run_my_scenario_test.py --config configs/my_scenario.yaml --duration 120 --follow
"""

import argparse
import math
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import DEFAULT_SCENARIO_CONFIG, load_runtime_config, validate_python_runtime

add_carla_pythonapi()

import carla
from scenarios_impl.my_scenario import build_scenario
from scenarios_impl.route_tracker import RouteTracker


def load_config(config_path):
    return load_runtime_config(scenario_path=config_path)


def spawn_ego(world, cfg):
    bp_lib = world.get_blueprint_library()
    ego_cfg = cfg.get("ego", {})
    vehicle_type = ego_cfg.get("vehicle_type", "vehicle.tesla.model3")
    spawn = ego_cfg.get("spawn_point", {})

    x = float(spawn.get("x", 0.0))
    y = float(spawn.get("y", 0.0))
    z = float(spawn.get("z", 0.5))
    yaw = float(spawn.get("yaw", 0.0))

    bp = bp_lib.find(vehicle_type)
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw),
    )

    ego = world.try_spawn_actor(bp, transform)
    if ego is None:
        raise RuntimeError(f"无法生成 ego @ ({x}, {y}, {z})")

    world.wait_for_tick()
    print(f"[TEST] 已生成 ego: {ego.type_id} @ {ego.get_location()}")
    return ego


def compute_steer_to_location(ego, target_loc, gain=1.25, max_steer=0.46):
    """算车辆到目标点的转向角。"""
    ego_loc = ego.get_location()
    ego_yaw = math.radians(ego.get_transform().rotation.yaw)

    dx = target_loc.x - ego_loc.x
    dy = target_loc.y - ego_loc.y
    target_yaw = math.atan2(dy, dx)

    delta = target_yaw - ego_yaw
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi

    steer = gain * delta
    steer = max(-max_steer, min(max_steer, steer))
    return steer


def route_following_controller(ego, route_tracker, progress_m, state, lookahead_m=12.0):
    """路线跟随控制器 + 换道。不做红绿灯刹车。"""
    # 换道目标横向偏移
    lateral_offset = float(state.get("lane_change_target", 0.0))

    # 路线跟随
    target_loc = route_tracker.point_at_progress_smoothed(
        progress_m + lookahead_m,
        window_radius_m=8.0,
        sample_step_m=2.0,
        sigma_m=4.0,
    )

    # 如果有换道，目标点加横向偏移
    if abs(lateral_offset) > 0.1:
        ego_yaw = math.radians(ego.get_transform().rotation.yaw)
        right = carla.Vector3D(
            x=math.cos(ego_yaw + math.pi / 2),
            y=math.sin(ego_yaw + math.pi / 2),
            z=0.0)
        target_loc = carla.Location(
            x=target_loc.x + right.x * lateral_offset,
            y=target_loc.y + right.y * lateral_offset,
            z=target_loc.z,
        )

    steer = compute_steer_to_location(ego, target_loc, gain=1.25, max_steer=0.46)

    # 速度控制
    v = ego.get_velocity()
    speed_kmh = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
    target_speed = float(state.get("target_speed_kmh", 50.0))
    if state.get("emergency_brake"):
        return 0.0, 1.0, steer

    if speed_kmh < target_speed - 5:
        throttle, brake = 0.5, 0.0
    elif speed_kmh > target_speed + 5:
        throttle, brake = 0.0, 0.2
    else:
        throttle, brake = 0.2, 0.0

    if abs(steer) > 0.3:
        throttle = min(throttle, 0.3)

    return throttle, brake, steer


def apply_simple_fallback_policy(state, progress_m):
    """Use scenario interfaces to keep the demo controller from driving into actors."""
    scenario_obs = state.get("scenario_observation", {})
    static_obstacles = scenario_obs.get("static_obstacles", [])
    dynamic_obstacles = scenario_obs.get("dynamic_obstacles", [])
    pedestrians = scenario_obs.get("pedestrians", [])
    active_events = scenario_obs.get("active_events", [])

    state["emergency_brake"] = False
    state["target_speed_kmh"] = 50.0

    if state.get("avoid_until_progress_m", 0.0) < progress_m:
        state["lane_change_target"] = 0.0

    nearest_static = min(
        (obs.get("distance_m", 999.0) for obs in static_obstacles),
        default=999.0,
    )
    nearest_dynamic = min(
        (obs.get("distance_m", 999.0) for obs in dynamic_obstacles),
        default=999.0,
    )
    nearest_pedestrian = min(
        (obs.get("distance_m", 999.0) for obs in pedestrians),
        default=999.0,
    )

    if nearest_static < 42.0:
        state["lane_change_target"] = -3.4
        state["avoid_until_progress_m"] = progress_m + 80.0
        state["target_speed_kmh"] = 22.0
    elif nearest_static < 60.0:
        state["target_speed_kmh"] = 28.0

    if nearest_dynamic < 12.0:
        state["emergency_brake"] = True
    elif nearest_dynamic < 28.0:
        state["target_speed_kmh"] = min(float(state.get("target_speed_kmh", 50.0)), 18.0)

    if nearest_pedestrian < 14.0:
        state["emergency_brake"] = True
    elif nearest_pedestrian < 30.0:
        state["target_speed_kmh"] = min(float(state.get("target_speed_kmh", 50.0)), 12.0)

    if any(event.get("task") == "final_stop" for event in active_events):
        state["target_speed_kmh"] = min(float(state.get("target_speed_kmh", 50.0)), 8.0)


def move_spectator_to_ego(spectator, ego):
    tf = ego.get_transform()
    spectator.set_transform(carla.Transform(
        tf.location + carla.Location(z=15, x=-8),
        carla.Rotation(pitch=-25, yaw=tf.rotation.yaw),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--manual", action="store_true",
                        help="只运行场景触发，不使用内置测试控制器控制 ego")
    parser.add_argument("--follow", action="store_true",
                        help="让观察者跟随 ego（默认不跟随，你可以自由移动视角）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    system_cfg = cfg["_system"]
    validate_python_runtime(system_cfg)
    runtime_cfg = system_cfg["runtime"]
    if args.duration == 120.0:
        args.duration = float(runtime_cfg["max_duration_seconds"])
    print(f"[TEST] 已加载配置: {args.config}")

    client = carla.Client(
        args.host or system_cfg["carla"]["host"],
        int(args.port or system_cfg["carla"]["port"]),
    )
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"[TEST] 已连接 CARLA")

    settings = world.get_settings()
    settings.synchronous_mode = bool(runtime_cfg["synchronous_mode"])
    settings.fixed_delta_seconds = float(runtime_cfg["fixed_delta_seconds"])
    world.apply_settings(settings)
    sync_mode = settings.synchronous_mode
    print(f"[DEBUG] 同步模式: {sync_mode}, 固定步长: {settings.fixed_delta_seconds}")

    # 1. 生成 ego
    ego = spawn_ego(world, cfg)

    # 2. 创建 RouteTracker（加载路线，用 GlobalRoutePlanner 补全）
    route_xml = cfg.get("route", {}).get("route_file", "routes/my_scenario.xml")
    route_tracker = RouteTracker(world, route_xml, sampling_resolution=2.0)

    # 3. 生成 NPC
    scenario = build_scenario()
    actors = scenario.spawn_actors(world, ego, cfg)
    print(f"[TEST] 已生成 {len(actors)} 个 NPC")

    # 4. 观察者
    spectator = world.get_spectator()
    move_spectator_to_ego(spectator, ego)
    if args.follow:
        print(f"[TEST] 观察者跟随模式")
    else:
        print(f"[TEST] 观察者自由模式")
        print(f"[TEST] 你可以在 CARLA 窗口里用 WASD 移动、鼠标右键拖动旋转、滚轮缩放")
        print(f"[TEST] ego 出生点: {ego.get_location()}")

    # 5. 主循环
    print(f"[TEST] 开始运行 {args.duration} 秒...")
    clock = settings.fixed_delta_seconds
    start_time = time.time()
    state = {}
    frame_count = 0
    last_print = 0

    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time

            # 算进度（单调递增）
            ego_loc = ego.get_location()
            progress_m = route_tracker.nearest_progress(ego_loc)

            if elapsed - last_print >= 5.0:
                last_print = elapsed
                v = ego.get_velocity()
                speed_kmh = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
                print(f"[TEST] t={elapsed:.1f}s, 帧={frame_count}, "
                      f"位置=({ego_loc.x:.1f},{ego_loc.y:.1f}), "
                      f"速度={speed_kmh:.1f}km/h, "
                      f"进度={progress_m:.1f}m")

            if args.follow:
                move_spectator_to_ego(spectator, ego)

            # 读状态
            obs = {
                "speed_kmh": math.sqrt(
                    ego.get_velocity().x ** 2 +
                    ego.get_velocity().y ** 2 +
                    ego.get_velocity().z ** 2
                ) * 3.6,
                "location": ego_loc,
                "route_metrics": {
                    "route_progress_m": progress_m,
                    "max_route_progress_m": route_tracker._max_progress,
                },
            }

            # 更新场景状态
            state = scenario.update_state(ego, actors, state, obs, clock, cfg)
            apply_simple_fallback_policy(state, progress_m)

            if not args.manual:
                # 路线跟随控制
                throttle, brake, steer = route_following_controller(
                    ego, route_tracker, progress_m, state,lookahead_m=12.0
                )

                # 应用控制
                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                ego.apply_control(control)

            if sync_mode:
                world.tick()
            else:
                world.wait_for_tick()

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[TEST] 用户中断")
    finally:
        print(f"[TEST] 共运行 {frame_count} 帧")
        print("[TEST] 清理 NPC...")
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        print("[TEST] 清理 ego...")
        if ego is not None and ego.is_alive:
            ego.destroy()

    print("[TEST] 场景测试完成")


if __name__ == "__main__":
    main()
