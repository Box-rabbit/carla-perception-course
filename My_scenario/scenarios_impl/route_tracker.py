"""RouteTracker：用 GlobalRoutePlanner 补全路线，按进度查位置。"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.carla import add_carla_pythonapi

add_carla_pythonapi()

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


def _extract_waypoint(item):
    """兼容两种 trace_route 返回格式：
    - carla.Waypoint 对象
    - (carla.Waypoint, RoadOption) 元组
    """
    if isinstance(item, tuple):
        return item[0]
    return item


class RouteTracker:
    """管理路线，提供按进度查位置的功能。"""

    def __init__(self, world, route_xml_path, sampling_resolution=2.0):
        self._world = world
        self._map = world.get_map()
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution)

        # 1. 读 XML 里的锚点
        anchors = self._load_anchors(route_xml_path)
        print(f"[RouteTracker] 锚点数量: {len(anchors)}")

        # 2. 用 GlobalRoutePlanner 补全
        self._route = self._build_route(anchors)
        print(f"[RouteTracker] 补全后路径点数量: {len(self._route)}")

        # 3. 算累计进度
        self._progresses = [0.0]
        for first, second in zip(self._route, self._route[1:]):
            wp_first = _extract_waypoint(first)
            wp_second = _extract_waypoint(second)
            self._progresses.append(
                self._progresses[-1]
                + wp_first.transform.location.distance(wp_second.transform.location)
            )
        self.route_total_length_m = self._progresses[-1]
        print(f"[RouteTracker] 路线总长度: {self.route_total_length_m:.1f} m")

        # 4. 单调递增用的状态
        self._last_index = 0
        self._max_progress = 0.0

    def _load_anchors(self, xml_path):
        root = ET.parse(xml_path).getroot()
        route = root.find('route')
        anchors = []
        for wp in route.findall('waypoint'):
            x = float(wp.get('x'))
            y = float(wp.get('y'))
            z = float(wp.get('z', 0.5))
            anchors.append(carla.Location(x=x, y=y, z=z))
        return anchors

    def _build_route(self, anchors):
        route = []
        for i in range(len(anchors) - 1):
            start = anchors[i]
            end = anchors[i + 1]
            segment = self._grp.trace_route(start, end)
            if i > 0:
                segment = segment[1:]

            # ⭐ 打印每一段
            seg_len = 0.0
            for a, b in zip(segment, segment[1:]):
                wp_a = _extract_waypoint(a)
                wp_b = _extract_waypoint(b)
                seg_len += wp_a.transform.location.distance(wp_b.transform.location)

            print(f"[RouteTracker] 段 {i:2d}→{i+1:2d}: "
                f"({start.x:7.1f},{start.y:7.1f}) → ({end.x:7.1f},{end.y:7.1f}), "
                f"{len(segment):4d} 点, {seg_len:7.1f}m")

            route.extend(segment)
        return route

    def point_at_progress(self, progress_m):
        """给定进度，返回路线上的坐标。"""
        if not self._route:
            return self._world.get_map().get_spawn_points()[0].location

        progress_m = max(0.0, min(progress_m, self.route_total_length_m))

        # 二分查找
        lo, hi = 0, len(self._progresses) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._progresses[mid] < progress_m:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx > 0 and abs(self._progresses[idx - 1] - progress_m) < abs(self._progresses[idx] - progress_m):
            idx = idx - 1

        wp = _extract_waypoint(self._route[idx])
        return wp.transform.location

    def point_at_progress_smoothed(self, progress_m, window_radius_m=8.0,
                                     sample_step_m=2.0, sigma_m=4.0):
        """平滑版：在目标点附近采样，加权平均，避免转向抖动。"""
        if not self._route:
            return self._world.get_map().get_spawn_points()[0].location

        samples = []
        for offset in range(-int(window_radius_m), int(window_radius_m) + 1, int(sample_step_m)):
            p = progress_m + offset
            if 0.0 <= p <= self.route_total_length_m:
                loc = self.point_at_progress(p)
                weight = math.exp(-(offset ** 2) / (2.0 * sigma_m ** 2))
                samples.append((loc, weight))

        if not samples:
            return self.point_at_progress(progress_m)

        total_weight = sum(w for _, w in samples)
        if total_weight <= 0.0:
            return self.point_at_progress(progress_m)

        x = sum(loc.x * w for loc, w in samples) / total_weight
        y = sum(loc.y * w for loc, w in samples) / total_weight
        z = sum(loc.z * w for loc, w in samples) / total_weight
        return carla.Location(x=x, y=y, z=z)

    def nearest_progress(self, location):
        """给定坐标，返回最近的路线进度（单调递增，不回头）。"""
        if not self._route:
            return 0.0

        # 从上次的 index 往前搜，最多搜 200 个点
        start = self._last_index
        end = min(len(self._route), start + 200)

        best_idx = start
        best_dist = float('inf')
        for i in range(start, end):
            wp = _extract_waypoint(self._route[i])
            d = wp.transform.location.distance(location)
            if d < best_dist:
                best_dist = d
                best_idx = i

        # 更新 last_index
        self._last_index = max(self._last_index, best_idx)
        progress = self._progresses[self._last_index]
        self._max_progress = max(self._max_progress, progress)
        return progress
