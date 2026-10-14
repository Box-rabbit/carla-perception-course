"""验证 My_scenario 的文件格式。"""

import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

SCENARIO_ROOT = Path(__file__).resolve().parent

# 验证路线 XML
print("=" * 50)
print("验证路线 XML")
print("=" * 50)
tree = ET.parse(SCENARIO_ROOT / 'routes' / 'my_scenario.xml')
root = tree.getroot()
routes = root.findall('route')
print(f"路线数量: {len(routes)}")
for r in routes:
    print(f"  ID: {r.get('id')}")
    print(f"  Town: {r.get('town')}")
    print(f"  路径点数量: {len(r.findall('waypoint'))}")
    print(f"  场景数量: {len(r.findall('.//scenario'))}")

# 验证配置 YAML
print()
print("=" * 50)
print("验证配置 YAML")
print("=" * 50)
with (SCENARIO_ROOT / 'configs' / 'my_scenario.yaml').open(encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(f"scenario_id: {cfg.get('scenario_id')}")
print(f"town: {cfg.get('map', {}).get('town')}")
print(f"ego 出生点: {cfg.get('ego', {}).get('spawn_point')}")
print(f"行人数量: {len(cfg.get('actors', {}).get('pedestrians', []))}")
print(f"车辆数量: {len(cfg.get('actors', {}).get('vehicles', []))}")

print()
print("=" * 50)
print("验证完成")
print("=" * 50)
