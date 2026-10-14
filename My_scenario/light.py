"""找出路线上的红绿灯。"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import load_system_config

add_carla_pythonapi()

sys.path.insert(0, str(Path(__file__).parent))

import carla
from scenarios_impl.route_tracker import RouteTracker


def main():
    system = load_system_config()
    client = carla.Client(system["carla"]["host"], int(system["carla"]["port"]))
    client.set_timeout(20.0)
    world = client.get_world()

    route_tracker = RouteTracker(
        world,
        str(REPOSITORY_ROOT / "My_scenario" / "routes" / "my_scenario.xml"),
        sampling_resolution=2.0,
    )

    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    print()
    print("路线上 30 米内的红绿灯:")
    print("=" * 60)

    seen_ids = set()
    for progress in range(0, int(route_tracker.route_total_length_m), 10):
        loc = route_tracker.point_at_progress(progress)
        for tl in traffic_lights:
            tl_loc = tl.get_location()
            d = tl_loc.distance(loc)
            if d <= 30.0 and tl.id not in seen_ids:
                seen_ids.add(tl.id)
                print(f"  进度 {progress:5.0f}m: "
                      f"红绿灯 id={tl.id}, "
                      f"位置=({tl_loc.x:.1f}, {tl_loc.y:.1f}), "
                      f"距离={d:.1f}m")

    print()
    print(f"路线上一共涉及 {len(seen_ids)} 个红绿灯")


if __name__ == "__main__":
    main()
