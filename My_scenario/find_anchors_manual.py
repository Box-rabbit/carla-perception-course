"""手动找锚点 + 自动吸附到车道中心。

操作：
1. 在 CARLA 窗口里用 WASD 飞到你要的位置
2. 按 Enter 键记录当前位置
3. 脚本会自动把坐标吸附到最近的车道中心
4. 记录完所有锚点后，按 Ctrl+C 退出
"""

import time
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi
from common.config import load_system_config

add_carla_pythonapi()
import carla


def snap_to_lane(carla_map, location):
    """把坐标吸附到最近的车道中心。"""
    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return None
    return waypoint.transform.location


def main():
    system = load_system_config()
    client = carla.Client(system["carla"]["host"], int(system["carla"]["port"]))
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    spectator = world.get_spectator()

    print("=" * 60)
    print("手动找锚点工具")
    print("=" * 60)
    print("操作:")
    print("  1. 在 CARLA 窗口里用 WASD 飞到目标位置")
    print("  2. 回到这个窗口，按 Enter 记录")
    print("  3. 脚本会自动吸附到最近的车道中心")
    print("  4. 记录完所有锚点后，按 Ctrl+C 退出")
    print("=" * 60)
    print()

    anchors = []

    try:
        while True:
            input("按 Enter 记录当前位置...")
            loc = spectator.get_transform().location
            print(f"  观察者位置: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")

            # 吸附到车道中心
            snapped = snap_to_lane(carla_map, loc)
            if snapped is None:
                print(f"  [警告] 找不到最近的车道，跳过")
                continue

            anchors.append(snapped)
            print(f"  吸附后坐标: x={snapped.x:.2f}, y={snapped.y:.2f}, z={snapped.z:.2f}")
            print(f"  已记录 {len(anchors)} 个锚点")
            print()

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print(f"共记录 {len(anchors)} 个锚点:")
        print("=" * 60)
        for i, loc in enumerate(anchors):
            print(f'  <waypoint x="{loc.x:.2f}" y="{loc.y:.2f}" z="{loc.z:.2f}" />')
        print()
        print("把这些 waypoint 复制到 routes/my_scenario.xml 里")


if __name__ == "__main__":
    main()
