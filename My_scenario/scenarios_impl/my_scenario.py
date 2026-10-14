"""Lightweight route-triggered scenarios for the project.

The YAML file describes when and what to trigger. This module only builds the
world-side test situation and exposes interfaces for perception/planning/control.
It does not attach sensors and does not command the ego vehicle.
"""

import math

import carla


def _state_name(tl_state):
    if tl_state == carla.TrafficLightState.Red:
        return "red"
    if tl_state == carla.TrafficLightState.Yellow:
        return "yellow"
    if tl_state == carla.TrafficLightState.Green:
        return "green"
    if tl_state == carla.TrafficLightState.Off:
        return "off"
    return "unknown"


class MyScenario:
    """Scenario implementation with route-distance triggers."""

    def __init__(self):
        self._world = None
        self._map = None
        self._pedestrians = []
        self._vehicles = []
        self._static_props = []
        self._traffic_lights = {}
        self._active_events = {}
        self._actors = []

    # ============================================================
    # Actor setup
    # ============================================================
    def spawn_actors(self, world, ego, cfg):
        """Pre-spawn scenario actors offstage. Ego and sensors are external."""
        self._world = world
        self._map = world.get_map()
        actors = []
        self._actors = actors
        bp_lib = world.get_blueprint_library()
        offstage = carla.Transform(carla.Location(x=10000.0, y=10000.0, z=-50.0))

        for instr in cfg.get("instructions", []):
            task = instr.get("task", "")
            if task in ("crossing_pedestrian", "pedestrian_crossing"):
                actor = self._spawn_blueprint_actor(
                    bp_lib,
                    world,
                    instr.get("pedestrian", {}),
                    "walker.pedestrian.0001",
                    "walker.pedestrian.*",
                    offstage,
                )
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actors.append(actor)
                    self._pedestrians.append({
                        "actor": actor,
                        "instr": instr,
                        "active": False,
                        "target": None,
                    })

                if "blocker" in instr:
                    self._vehicles.append({
                        "actor": None,
                        "actor_cfg": instr.get("blocker", {}),
                        "instr": instr,
                        "role": "pedestrian_blocker",
                        "active": False,
                        "target_speed_kmh": 0.0,
                    })

            elif task in ("blocked_lane", "lead_vehicle", "merge_conflict",
                          "junction_conflict", "overtake"):
                cfg_key = {
                    "blocked_lane": "obstacle",
                    "lead_vehicle": "lead_vehicle",
                    "merge_conflict": "conflict_vehicle",
                    "junction_conflict": "conflict_vehicle",
                    "overtake": "target_vehicle",
                }.get(task, "vehicle")
                self._vehicles.append({
                    "actor": None,
                    "actor_cfg": instr.get(cfg_key, {}),
                    "instr": instr,
                    "role": task,
                    "active": False,
                    "target_speed_kmh": 0.0,
                })

        self._configure_traffic_lights(world, cfg)
        print(f"[MyScenario] spawned {len(actors)} scenario actors")
        return actors

    def _spawn_blueprint_actor(self, bp_lib, world, actor_cfg, default_model,
                               default_filter, transform):
        model = actor_cfg.get("model", default_model)
        fallback = actor_cfg.get("fallback_filter", default_filter)
        try:
            bp = bp_lib.find(model)
        except Exception:
            matches = bp_lib.filter(fallback)
            if not matches:
                print(f"[MyScenario][WARN] no blueprint for {model}, skipped")
                return None
            bp = matches[0]

        actor = world.try_spawn_actor(bp, transform)
        if actor is None:
            print(f"[MyScenario][WARN] failed to spawn {bp.id}")
        else:
            print(f"[MyScenario] spawned {bp.id} for scenario actor id={actor.id}")
        return actor

    def _configure_traffic_lights(self, world, cfg):
        """Set global light durations and expose light actors for perception eval."""
        tl_cfg = cfg.get("traffic_lights", {})
        cycle = tl_cfg.get("cycle", {})
        green_time = float(cycle.get("green_time_s", 20.0))
        red_time = float(cycle.get("red_time_s", 20.0))
        yellow_time = float(cycle.get("yellow_time_s", 3.0))

        for tl in world.get_actors().filter("traffic.traffic_light"):
            tl.freeze(False)
            if tl.get_state() == carla.TrafficLightState.Off:
                tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(green_time)
            tl.set_red_time(red_time)
            tl.set_yellow_time(yellow_time)
            self._traffic_lights[tl.id] = tl

        print(
            "[MyScenario] traffic light cycle set for "
            f"{len(self._traffic_lights)} lights: "
            f"green={green_time}s red={red_time}s yellow={yellow_time}s"
        )

    # ============================================================
    # Per-frame update
    # ============================================================
    def update_state(self, ego, actors, state, obs, dt, cfg):
        """Trigger configured scenarios and update non-ego actors."""
        route_metrics = obs.get("route_metrics", {})
        progress = float(route_metrics.get("route_progress_m", 0.0))
        max_progress = float(route_metrics.get("max_route_progress_m", progress))

        for instr in cfg.get("instructions", []):
            instr_id = instr.get("id", "")
            trigger = instr.get("trigger", {})
            trigger_progress = float(trigger.get("value", 0.0))

            if state.get(f"{instr_id}_triggered"):
                continue
            if max_progress < trigger_progress:
                continue

            state[f"{instr_id}_triggered"] = True
            print(
                f"[MyScenario] trigger {instr_id} at {progress:.1f}m "
                f"(task={instr.get('task', '')})"
            )
            self._dispatch_scenario(instr, ego, state)

        self._update_pedestrians(dt)
        self._update_vehicles()
        self._expire_events(progress)
        self._update_interfaces(state, obs, ego, progress, cfg)
        return state

    def _dispatch_scenario(self, instr, ego, state):
        task = instr.get("task", "")
        if task in ("crossing_pedestrian", "pedestrian_crossing"):
            self._on_crossing_pedestrian(ego, instr, state)
        elif task == "blocked_lane":
            self._on_blocked_lane(ego, instr, state)
        elif task in ("lead_vehicle", "overtake"):
            self._on_lead_vehicle(ego, instr, state)
        elif task == "merge_conflict":
            self._on_merge_conflict(ego, instr, state)
        elif task in ("junction_conflict", "right_turn_no_light", "merge"):
            self._on_junction_conflict(ego, instr, state)
        elif task == "final_stop":
            self._on_final_stop(instr, state)
        elif task == "traffic_light_monitor":
            self._record_event(instr, "traffic_light_monitor", "active", state)
        else:
            print(f"[MyScenario][WARN] unknown task: {task}")

    # ============================================================
    # Scenario handlers
    # ============================================================
    def _on_crossing_pedestrian(self, ego, instr, state):
        for ped in self._pedestrians:
            if ped["instr"].get("id") != instr.get("id"):
                continue

            ped_cfg = instr.get("pedestrian", {})
            distance = float(ped_cfg.get("distance_ahead_m", 18.0))
            start_lateral = float(ped_cfg.get("start_lateral_offset_m", 5.0))
            end_lateral = float(ped_cfg.get("end_lateral_offset_m", -5.0))
            speed = float(ped_cfg.get("speed_mps", 1.4))

            start_loc = self._relative_location(ego, distance, start_lateral, 0.8)
            end_loc = self._relative_location(ego, distance, end_lateral, 0.8)
            ground_z = self._road_z_at(self._relative_location(ego, distance, 0.0, 0.0))
            if ground_z is not None:
                start_loc.z = ground_z + 1.0
                end_loc.z = ground_z + 1.0
            yaw = ego.get_transform().rotation.yaw + 90.0
            ped["actor"].set_transform(carla.Transform(start_loc, carla.Rotation(yaw=yaw)))
            ped["actor"].set_simulate_physics(True)
            ped["target"] = end_loc
            ped["speed_mps"] = speed
            ped["active"] = True

            self._record_event(instr, "pedestrian", "active", state)
            self._activate_pedestrian_blocker(ego, instr)
            print(
                f"[MyScenario] pedestrian crossing active: {instr.get('id')} "
                f"start=({start_loc.x:.1f},{start_loc.y:.1f},{start_loc.z:.1f}) "
                f"end=({end_loc.x:.1f},{end_loc.y:.1f},{end_loc.z:.1f})"
            )
            return

        print(f"[MyScenario][WARN] no pedestrian actor for {instr.get('id')}")

    def _activate_pedestrian_blocker(self, ego, instr):
        blocker_cfg = instr.get("blocker")
        if not blocker_cfg:
            return

        veh = self._find_vehicle(instr)
        if veh is None:
            return

        distance = float(blocker_cfg.get("distance_ahead_m", 14.0))
        lateral = float(blocker_cfg.get("lateral_offset_m", 3.2))
        yaw_delta = float(blocker_cfg.get("yaw_delta_deg", 0.0))
        transform = self._relative_transform(ego, distance, lateral, 0.5, yaw_delta)

        actor = self._ensure_vehicle_actor(veh, transform)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        veh.update({"active": True, "target_speed_kmh": 0.0, "hold_brake": True})

    def _on_blocked_lane(self, ego, instr, state):
        veh = self._find_vehicle(instr)
        if veh is None:
            return

        obstacle_cfg = instr.get("obstacle", {})
        distance = float(obstacle_cfg.get("distance_ahead_m", 45.0))
        lateral = float(obstacle_cfg.get("lateral_offset_m", 0.0))
        transform = self._relative_transform(ego, distance, lateral, 0.5, 0.0)
        actor = self._ensure_vehicle_actor(veh, transform)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        veh.update({"active": True, "target_speed_kmh": 0.0, "hold_brake": True})

        self._record_event(instr, "static_obstacle", "active", state)
        print(f"[MyScenario] blocked lane obstacle active: {instr.get('id')}")

    def _on_lead_vehicle(self, ego, instr, state):
        veh = self._find_vehicle(instr)
        if veh is None:
            return

        lead_cfg = instr.get("lead_vehicle", instr.get("target_vehicle", {}))
        distance = float(lead_cfg.get("distance_ahead_m", 30.0))
        lateral = float(lead_cfg.get("lateral_offset_m", 0.0))
        speed_kmh = float(lead_cfg.get("speed_kmh", 12.0))
        hand_brake = bool(lead_cfg.get("hand_brake", False)) or speed_kmh <= 0.1
        transform = self._relative_transform(ego, distance, lateral, 0.5, 0.0)

        actor = self._ensure_vehicle_actor(veh, transform)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        if hand_brake:
            actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        veh.update({
            "active": True,
            "target_speed_kmh": speed_kmh,
            "hold_brake": hand_brake,
        })

        event_type = "static_obstacle" if hand_brake else "lead_vehicle"
        self._record_event(instr, event_type, "active", state)
        print(
            f"[MyScenario] lead vehicle active: {instr.get('id')} "
            f"speed={speed_kmh}km/h hand_brake={hand_brake}"
        )

    def _on_merge_conflict(self, ego, instr, state):
        veh = self._find_vehicle(instr)
        if veh is None:
            return

        cfg = instr.get("conflict_vehicle", instr.get("merge_vehicle", {}))
        distance = float(cfg.get("distance_ahead_m", 35.0))
        lateral = float(cfg.get("lateral_offset_m", -7.0))
        yaw_delta = float(cfg.get("yaw_delta_deg", 0.0))
        speed_kmh = float(cfg.get("speed_kmh", 22.0))
        follow_waypoints = bool(cfg.get("follow_waypoints", True))
        destroy_after_progress_m = float(cfg.get("destroy_after_progress_m", 80.0))
        if cfg.get("placement") == "left_side_road":
            transform = self._side_road_transform(
                ego, distance, lateral, z_offset=0.5, yaw_delta=yaw_delta
            )
        elif cfg.get("placement") == "left_adjacent_lane":
            transform = self._adjacent_lane_transform(
                ego, distance, side="left", z_offset=0.5, yaw_delta=yaw_delta
            )
        else:
            transform = self._relative_transform(ego, distance, lateral, 0.5, yaw_delta)
        if follow_waypoints:
            transform = self._project_transform_to_driving_lane(transform)

        actor = self._ensure_vehicle_actor(veh, transform)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        veh.update({
            "active": True,
            "target_speed_kmh": speed_kmh,
            "hold_brake": False,
            "follow_waypoints": follow_waypoints,
            "destroy_after_progress_m": destroy_after_progress_m,
            "start_location": transform.location,
        })

        self._record_event(instr, "merge_conflict", "active", state)
        loc = transform.location
        print(
            f"[MyScenario] merge conflict vehicle active: {instr.get('id')} "
            f"spawn=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}) "
            f"yaw={transform.rotation.yaw:.1f}"
        )

    def _on_junction_conflict(self, ego, instr, state):
        if instr.get("disable_traffic_lights", False):
            self._disable_nearby_traffic_lights(ego, instr)

        veh = self._find_vehicle(instr)
        if veh is None:
            self._record_event(instr, "junction_conflict", "active", state)
            print(f"[MyScenario] junction conflict marker active: {instr.get('id')}")
            return

        cfg = instr.get("conflict_vehicle", {})
        distance = float(cfg.get("distance_ahead_m", 25.0))
        lateral = float(cfg.get("lateral_offset_m", 10.0))
        yaw_delta = float(cfg.get("yaw_delta_deg", -90.0))
        speed_kmh = float(cfg.get("speed_kmh", 20.0))
        follow_waypoints = bool(cfg.get("follow_waypoints", True))
        destroy_after_progress_m = float(cfg.get("destroy_after_progress_m", 70.0))
        transform = self._relative_transform(ego, distance, lateral, 0.5, yaw_delta)
        if follow_waypoints:
            transform = self._project_transform_to_driving_lane(transform)

        actor = self._ensure_vehicle_actor(veh, transform)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        veh.update({
            "active": True,
            "target_speed_kmh": speed_kmh,
            "hold_brake": False,
            "follow_waypoints": follow_waypoints,
            "destroy_after_progress_m": destroy_after_progress_m,
            "start_location": transform.location,
        })

        self._record_event(instr, "junction_conflict", "active", state)
        print(f"[MyScenario] junction conflict vehicle active: {instr.get('id')}")

    def _on_final_stop(self, instr, state):
        self._record_event(instr, "final_stop", "active", state)
        print(f"[MyScenario] final stop area active: {instr.get('id')}")

    # ============================================================
    # Actor updates
    # ============================================================
    def _update_pedestrians(self, dt):
        for ped in self._pedestrians:
            if not ped.get("active"):
                continue
            actor = ped["actor"]
            target = ped.get("target")
            if target is None:
                continue

            current = actor.get_location()
            dx = target.x - current.x
            dy = target.y - current.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.5:
                ped["active"] = False
                continue

            direction = carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0)
            speed = float(ped.get("speed_mps", 1.4))
            actor.apply_control(carla.WalkerControl(direction=direction, speed=speed))

    def _update_vehicles(self):
        for veh in self._vehicles:
            if not veh.get("active"):
                continue

            actor = veh["actor"]
            if actor is None or not actor.is_alive:
                continue
            if self._maybe_destroy_finished_vehicle(veh):
                continue
            control = carla.VehicleControl()
            if veh.get("hold_brake"):
                control.brake = 1.0
                control.hand_brake = True
                actor.apply_control(control)
                continue

            target_speed = float(veh.get("target_speed_kmh", 0.0)) / 3.6
            if veh.get("follow_waypoints"):
                control = self._waypoint_vehicle_control(actor, target_speed)
                actor.apply_control(control)
                continue

            velocity = actor.get_velocity()
            current_speed = math.sqrt(
                velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
            )
            error = target_speed - current_speed
            if error > 0.5:
                control.throttle = min(0.65, error * 0.35)
            elif error < -0.5:
                control.brake = min(0.5, -error * 0.2)
            else:
                control.throttle = 0.2
            actor.apply_control(control)

    def _maybe_destroy_finished_vehicle(self, veh):
        if veh.get("hold_brake"):
            return False
        actor = veh.get("actor")
        start_location = veh.get("start_location")
        destroy_after = veh.get("destroy_after_progress_m")
        if actor is None or start_location is None or destroy_after is None:
            return False

        distance = actor.get_location().distance(start_location)
        if distance < float(destroy_after):
            return False

        actor_id = actor.id
        actor.destroy()
        veh["active"] = False
        veh["actor"] = None
        print(
            f"[MyScenario] destroyed dynamic vehicle for "
            f"{veh['instr'].get('id')} after {distance:.1f}m (actor id={actor_id})"
        )
        return True

    def _waypoint_vehicle_control(self, actor, target_speed):
        control = carla.VehicleControl()
        if self._map is None:
            control.throttle = 0.25
            return control

        actor_loc = actor.get_location()
        waypoint = self._map.get_waypoint(
            actor_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            control.brake = 0.5
            return control

        lookahead = max(4.0, min(10.0, target_speed * 1.2))
        next_wps = waypoint.next(lookahead)
        target_wp = next_wps[0] if next_wps else waypoint
        target_loc = target_wp.transform.location

        actor_tf = actor.get_transform()
        dx = target_loc.x - actor_tf.location.x
        dy = target_loc.y - actor_tf.location.y
        target_yaw = math.degrees(math.atan2(dy, dx))
        yaw_error = self._angle_delta(target_yaw, actor_tf.rotation.yaw)
        control.steer = max(-0.6, min(0.6, yaw_error / 45.0))

        velocity = actor.get_velocity()
        current_speed = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        speed_error = target_speed - current_speed
        if speed_error > 0.4:
            control.throttle = min(0.45, 0.18 + speed_error * 0.18)
        elif speed_error < -0.4:
            control.brake = min(0.4, -speed_error * 0.2)
        else:
            control.throttle = 0.12

        return control

    # ============================================================
    # Interfaces
    # ============================================================
    def _update_interfaces(self, state, obs, ego, progress, cfg):
        traffic_lights = self._traffic_light_observations(ego, cfg)
        actors = self._actor_observations(ego)
        active_events = list(self._active_events.values())

        state["scenario_observation"] = {
            "route_progress_m": progress,
            "active_events": active_events,
            "traffic_lights": traffic_lights,
            "dynamic_obstacles": actors["dynamic"],
            "static_obstacles": actors["static"],
            "pedestrians": actors["pedestrians"],
        }

        state["planning_interface"] = {
            "route_progress_m": progress,
            "active_events": active_events,
            "dynamic_obstacles": actors["dynamic"],
            "static_obstacles": actors["static"],
            "pedestrians": actors["pedestrians"],
        }

        state["control_interface"] = {
            "scenario_controls_ego": False,
            "fallback_hint_only": True,
        }

        # This is a scenario-side evaluation description.  It remains in the
        # truth/state channel and is never passed to PerceptionRuntime.  The
        # target rate is derived from the canonical system clock instead of
        # being duplicated in the scenario configuration.
        tl_cfg = cfg.get("traffic_lights", {})
        eval_cfg = tl_cfg.get("perception_eval", {})
        system_runtime = cfg.get("_system", {}).get("runtime", {})
        sensor_tick = float(system_runtime.get("sensor_tick_seconds", 0.05))
        state["perception_interface"] = {
            "camera_required": bool(eval_cfg.get("enabled", True)),
            "target_fps": (1.0 / sensor_tick) if sensor_tick > 0.0 else None,
            "traffic_light_detection_range_m": float(
                tl_cfg.get("detection_range_m", 50.0)
            ),
            "traffic_light_ground_truth": traffic_lights,
            "placeholder_result_source": "ground_truth",
        }

    def _traffic_light_observations(self, ego, cfg):
        detection_range = float(
            cfg.get("traffic_lights", {}).get("detection_range_m", 50.0)
        )
        ego_loc = ego.get_location()
        observations = []
        for tl_id, tl in self._traffic_lights.items():
            loc = tl.get_location()
            distance = ego_loc.distance(loc)
            if distance > detection_range:
                continue
            observations.append({
                "id": tl_id,
                "state": _state_name(tl.get_state()),
                "distance_m": distance,
                "location": {"x": loc.x, "y": loc.y, "z": loc.z},
            })
        observations.sort(key=lambda item: item["distance_m"])
        return observations

    def _actor_observations(self, ego):
        dynamic = []
        static = []
        pedestrians = []
        ego_loc = ego.get_location()

        for veh in self._vehicles:
            if not veh.get("active"):
                continue
            actor = veh["actor"]
            if actor is None or not actor.is_alive:
                continue
            loc = actor.get_location()
            item = {
                "id": actor.id,
                "scenario_id": veh["instr"].get("id"),
                "role": veh.get("role"),
                "distance_m": ego_loc.distance(loc),
                "location": {"x": loc.x, "y": loc.y, "z": loc.z},
                "target_speed_kmh": float(veh.get("target_speed_kmh", 0.0)),
            }
            if veh.get("hold_brake"):
                static.append(item)
            else:
                dynamic.append(item)

        for ped in self._pedestrians:
            if not ped.get("active"):
                continue
            actor = ped["actor"]
            loc = actor.get_location()
            pedestrians.append({
                "id": actor.id,
                "scenario_id": ped["instr"].get("id"),
                "distance_m": ego_loc.distance(loc),
                "location": {"x": loc.x, "y": loc.y, "z": loc.z},
                "speed_mps": float(ped.get("speed_mps", 1.4)),
            })

        return {"dynamic": dynamic, "static": static, "pedestrians": pedestrians}

    # ============================================================
    # Helpers
    # ============================================================
    def _find_vehicle(self, instr):
        instr_id = instr.get("id")
        for veh in self._vehicles:
            if veh["instr"].get("id") == instr_id:
                return veh
        print(f"[MyScenario][WARN] no vehicle actor for {instr_id}")
        return None

    def _ensure_vehicle_actor(self, veh, transform):
        actor = veh.get("actor")
        if actor is not None and actor.is_alive:
            actor.set_transform(transform)
            return actor

        if self._world is None:
            print("[MyScenario][WARN] world is not ready; cannot spawn vehicle")
            return None

        actor_cfg = veh.get("actor_cfg", {})
        bp_lib = self._world.get_blueprint_library()
        model = actor_cfg.get("model", "vehicle.audi.tt")
        fallback = actor_cfg.get("fallback_filter", "vehicle.*")

        blueprints = []
        try:
            blueprints.append(bp_lib.find(model))
        except Exception:
            pass
        blueprints.extend(bp for bp in bp_lib.filter(fallback) if bp not in blueprints)

        for bp in blueprints:
            actor = self._world.try_spawn_actor(bp, transform)
            if actor is not None:
                veh["actor"] = actor
                if actor not in self._actors:
                    self._actors.append(actor)
                print(
                    f"[MyScenario] spawned vehicle {bp.id} for "
                    f"{veh['instr'].get('id')} at trigger"
                )
                return actor

        loc = transform.location
        print(
            f"[MyScenario][WARN] failed to spawn vehicle for {veh['instr'].get('id')} "
            f"at ({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}) yaw={transform.rotation.yaw:.1f}"
        )
        return None

    def _disable_nearby_traffic_lights(self, ego, instr):
        cfg = instr.get("conflict_vehicle", {})
        center_distance = float(cfg.get("distance_ahead_m", 25.0))
        radius = float(instr.get("traffic_light_radius_m", 60.0))
        center = self._relative_location(ego, center_distance, 0.0, 0.0)
        disabled_ids = []

        for tl_id, tl in self._traffic_lights.items():
            if tl.get_location().distance(center) > radius:
                continue
            tl.set_state(carla.TrafficLightState.Off)
            tl.freeze(True)
            disabled_ids.append(tl_id)

        if disabled_ids:
            print(
                f"[MyScenario] disabled {len(disabled_ids)} traffic lights "
                f"near {instr.get('id')}: {disabled_ids}"
            )
        else:
            print(f"[MyScenario][WARN] no traffic lights found near {instr.get('id')}")

    def _project_transform_to_driving_lane(self, transform):
        if self._map is None:
            return transform
        waypoint = self._map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return transform
        projected = waypoint.transform
        projected.location.z += 0.5
        return projected

    def _record_event(self, instr, event_type, status, state):
        event = {
            "id": instr.get("id"),
            "task": instr.get("task"),
            "type": event_type,
            "status": status,
            "trigger": instr.get("trigger", {}),
            "duration_m": float(instr.get("scenario_duration_m", 120.0)),
        }
        self._active_events[instr.get("id")] = event
        state.setdefault("scenario_events", []).append(event.copy())

    def _expire_events(self, progress):
        for event_id, event in list(self._active_events.items()):
            trigger = event.get("trigger", {})
            start = float(trigger.get("value", progress))
            duration = float(event.get("duration_m", 120.0))
            if progress - start > duration:
                self._active_events.pop(event_id, None)

    def _side_road_transform(self, ego, forward_m, right_m, z_offset, yaw_delta):
        if self._map is None:
            return self._relative_transform(ego, forward_m, right_m, z_offset, yaw_delta)

        desired = self._relative_location(ego, forward_m, right_m, 0.0)
        ego_yaw = ego.get_transform().rotation.yaw
        target_yaw = ego_yaw + yaw_delta
        best_wp = None
        best_score = float("inf")

        # Search around the rough left-road point. The side road is perpendicular
        # to the route, so prefer driving waypoints whose yaw matches target_yaw.
        for dx in range(-20, 21, 4):
            for dy in range(-20, 21, 4):
                sample = carla.Location(x=desired.x + dx, y=desired.y + dy, z=desired.z)
                wp = self._map.get_waypoint(
                    sample,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if wp is None:
                    continue
                yaw_error = abs(self._angle_delta(wp.transform.rotation.yaw, target_yaw))
                if yaw_error > 50.0:
                    continue
                distance = wp.transform.location.distance(desired)
                if distance > 24.0:
                    continue
                score = distance + yaw_error * 0.2
                if score < best_score:
                    best_score = score
                    best_wp = wp

        if best_wp is None:
            print(
                "[MyScenario][WARN] no perpendicular side-road waypoint found, "
                "using projected relative placement"
            )
            projected = self._map.get_waypoint(
                desired,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if projected is not None:
                best_wp = projected
            else:
                return self._relative_transform(ego, forward_m, right_m, z_offset, yaw_delta)

        transform = best_wp.transform
        transform.location.z += z_offset
        # Keep the requested side-road direction when the search had to fall back.
        if abs(self._angle_delta(transform.rotation.yaw, target_yaw)) > 50.0:
            transform.rotation.yaw = target_yaw
        print(
            f"[MyScenario] side road placement: desired=({desired.x:.1f},{desired.y:.1f}) "
            f"wp=({transform.location.x:.1f},{transform.location.y:.1f}) "
            f"yaw={transform.rotation.yaw:.1f}"
        )
        return transform

    def _angle_delta(self, a, b):
        delta = a - b
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    def _adjacent_lane_transform(self, ego, forward_m, side, z_offset, yaw_delta):
        if self._map is None:
            return self._relative_transform(ego, forward_m, 0.0, z_offset, yaw_delta)

        reference_loc = self._relative_location(ego, forward_m, 0.0, 0.0)
        reference_wp = self._map.get_waypoint(
            reference_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if reference_wp is None:
            return self._relative_transform(ego, forward_m, 0.0, z_offset, yaw_delta)

        if side == "left":
            candidate = reference_wp.get_left_lane()
        else:
            candidate = reference_wp.get_right_lane()

        if not self._is_same_direction_driving_lane(reference_wp, candidate):
            print(
                f"[MyScenario][WARN] no same-direction {side} lane, "
                "falling back to route lane placement"
            )
            candidate = reference_wp

        transform = candidate.transform
        transform.location.z += z_offset
        transform.rotation.yaw += yaw_delta
        return transform

    def _is_same_direction_driving_lane(self, reference_wp, candidate_wp):
        if candidate_wp is None:
            return False
        if candidate_wp.lane_type != carla.LaneType.Driving:
            return False

        yaw_delta = abs(candidate_wp.transform.rotation.yaw - reference_wp.transform.rotation.yaw)
        while yaw_delta > 180.0:
            yaw_delta -= 360.0
        return abs(yaw_delta) < 45.0

    def _relative_transform(self, ego, forward_m, right_m, z_offset, yaw_delta):
        loc = self._relative_location(ego, forward_m, right_m, z_offset)
        yaw = ego.get_transform().rotation.yaw + yaw_delta
        return carla.Transform(loc, carla.Rotation(yaw=yaw))

    def _road_z_at(self, location):
        if self._map is None:
            return None
        waypoint = self._map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None
        return waypoint.transform.location.z

    def _relative_location(self, ego, forward_m, right_m, z_offset):
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        forward = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        return carla.Location(
            x=ego_loc.x + forward.x * forward_m + right.x * right_m,
            y=ego_loc.y + forward.y * forward_m + right.y * right_m,
            z=ego_loc.z + z_offset,
        )


def build_scenario():
    return MyScenario()
