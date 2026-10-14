# 数据链和帧契约

## 在线链路

```text
config/system.yaml + config/sensors.yaml
        |
        v
CARLA world settings + camera actors
        |
        | world.tick() -> frame_id
        v
CarlaSensorBridge: BGRA -> RGB, per-camera queue, frame pairing
        |
        v
SensorFrameBundle -> PerceptionRuntime -> perception.jsonl

CARLA snapshot + MyScenario state -----------------> truth.jsonl
```

`PerceptionRuntime.process(bundle)` 的唯一输入是 `SensorFrameBundle`。场景状态不在这个调用中出现。`truth.jsonl` 和 `perception.jsonl` 是两个独立文件，分别标记 `source: ground_truth` 和 `source: camera_model`（空处理器时为 `sensor_only`）。

## 每帧必须固定的字段

| 字段 | 定义位置 | 规则 |
| --- | --- | --- |
| `camera_id` | `config/sensors.yaml` 的 `sensors` 键 | 逻辑名称，不能使用 CARLA actor ID |
| `image_width_px`, `image_height_px` | `config/sensors.yaml` 的 `defaults.image_size_px`，相机可覆盖 | 回调尺寸必须完全匹配 |
| `sensor_tick_seconds` | `config/system.yaml` | 全部相机共用 `0.05`，不得在单相机重复定义 |
| `frame_id` | CARLA image 的 `image.frame` | 同一 `world.tick()` 的所有相机必须相同 |
| `timestamp_ns` | CARLA image 的 `image.timestamp` 转纳秒 | 同一 bundle 必须相同；来源为 `carla_simulation` |
| `frame_id_source` | `common/contracts.py` | 感知帧为 `carla_sensor`，真值帧为 `carla_snapshot` |
| `image_color_order` | `config/system.yaml` | `RGB`，由公共 BGRA 解码器保证 |
| 图像坐标系 | `config/system.yaml` | `image_pixels_x_right_y_down` |
| ego 坐标系 | `config/system.yaml` | `ego_vehicle_x_forward_y_right_z_up` |
| `episode_id` | 联调运行器生成 | 一次运行内保持不变 |

## 同步和丢帧规则

1. 开启同步模式，`fixed_delta_seconds` 必须等于 `sensor_tick_seconds`。
2. 先注册传感器回调，再调用 `world.tick()`。
3. 每次 tick 按返回的 frame ID 从每个启用相机取一帧。
4. 过期帧丢弃并计数；超前帧暂存，当前 bundle 报同步错误。
5. 缺帧超时、尺寸错误、时间戳不一致都不能进入感知处理器。
6. 感知输出禁止包含 `ground_truth`、`scenario_observation`、`traffic_light_ground_truth` 等真值字段。
