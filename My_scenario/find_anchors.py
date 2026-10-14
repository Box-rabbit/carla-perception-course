"""自动找环形路线的关键锚点。

思路：
1. 从起点出发，沿车道一直走
2. 每走一段，检测航向变化
3. 航向变化超过阈值的地方，就是转角
4. 把转角坐标打出来
"""

import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import load_system_config

add_carla_pythonapi()
import carla


def wrap_deg(delta):
    return (delta + 180.0) % 360.0 - 180.0


def main():
    system = load_system_config()
    client = carla.Client(system["carla"]["host"], int(system["carla"]["port"]))
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()

    # 起点
    start_loc = carla.Location(x=-20.178, y=-94.924, z=0.5)
    waypoint = carla_map.get_waypoint(start_loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if waypoint is None:
        print("找不到起点所在的车道")
        return

    print(f"起点: ({waypoint.transform.location.x:.2f}, "
          f"{waypoint.transform.location.y:.2f})")
    print()

    anchors = [waypoint.transform.location]
    current = waypoint
    last_yaw = current.transform.rotation.yaw
    total_distance = 0.0

    # 沿车道走，走 5 公里
    step = 5.0  # 每步 5 米
    max_distance = 5500.0

    while total_distance < max_distance:
        # 取前方 step 米的下一个 waypoint
        nexts = current.next(step)
        if not nexts:
            print("走到路尽头")
            break
        # 优先选直行
        current = nexts[0]
        total_distance += step

        # 检测航向变化
        new_yaw = current.transform.rotation.yaw
        delta = wrap_deg(new_yaw - last_yaw)

        # 航向变化超过 20 度，认为进入转弯
        if abs(delta) > 20.0:
            loc = current.transform.location
            print(f"[转弯] 进度={total_distance:.1f}m, "
                  f"航向变化={delta:+.1f}°, "
                  f"坐标=({loc.x:.2f}, {loc.y:.2f})")
            anchors.append(loc)
            # 跳到转弯结束
            last_yaw = new_yaw

        last_yaw = new_yaw

    print()
    print(f"共找到 {len(anchors)} 个锚点:")
    for i, loc in enumerate(anchors):
        print(f"  [{i}] x={loc.x:.2f}, y={loc.y:.2f}, z=0.5")


if __name__ == "__main__":
    main()
