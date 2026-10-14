# CARLA Perception Course Project

这是基于 CARLA 0.9.15 的课程设计仓库。项目将场景、传感器运行时、感知算法和评测数据分开管理。

## Python 环境

CARLA 0.9.15 的 PythonAPI 使用 Python 3.7。项目统一使用：

```bash
source scripts/project_env.sh
```

The 3.7 environment is also described in [`requirements-py37.txt`](requirements-py37.txt). OpenCV 4.6.0.66 and the CPU PyTorch stack are installed directly in this Python 3.7 environment.

`scripts/project_env.sh` only sets the project interpreter, CARLA path and CARLA egg cache; it does not inject packages from another Python installation.

## 数据链路

```text
CARLA world.tick()
    -> SensorFrame / SensorFrameBundle
    -> perception_runtime
    -> perception.jsonl (camera_model)

CARLA snapshot / actor state
    -> truth.jsonl (ground_truth, evaluation only)
```

场景真值和感知输入是两条独立通道。`PerceptionRuntime` 不接受场景状态参数，并拒绝在模型输出中出现 `ground_truth`、`scenario_observation` 或 `traffic_light_ground_truth` 字段。

## 目录边界

```text
config/                    # 唯一的公共运行时和传感器配置
common/                    # 场景和感知共用的配置/CARLA 工具
My_scenario/               # 场景、路线和真值
perception_runtime/        # 帧契约、同步队列、感知插件、日志
lane_detection/            # 项目自有 UFLD 封装和模型
traffic_light_classifier/  # 项目自有红绿灯检测/分类封装和模型
stereo_sgbm/               # 项目自有双目深度入口
calibration/               # 感知侧相机内外参交付目录
yolov5/                    # GitHub 上游代码和项目自有接口文件
ufld/                      # GitHub 上游代码
test_data/                 # 小型离线输入
work/                      # 合成测试素材
outputs/                   # 运行产物，不是配置来源
```

`yolov5/` 和 `ufld/` 内的 GitHub 上游文件保持不改。项目自有的帧接口文件放在 `yolov5/` 是历史兼容安排，新的在线入口统一使用 `perception_runtime/`。

## 唯一配置入口

- [`config/system.yaml`](config/system.yaml)：CARLA 地址、同步时钟、帧来源、坐标系和日志文件名。
- [`config/sensors.yaml`](config/sensors.yaml)：公共相机默认值、逻辑 `camera_id` 的差异项、安装位姿和标定文件引用。
- [`My_scenario/configs/my_scenario.yaml`](My_scenario/configs/my_scenario.yaml)：地图、ego、路线和场景触发器。

```python
from common.config import load_runtime_config

config = load_runtime_config()
```

不要在脚本中硬编码 CARLA 根路径或输出目录。

公共帧字段名和来源值集中在 [`common/contracts.py`](common/contracts.py)。代码中的 `camera_id`、图像分辨率、安装位姿和全局 `sensor_tick_seconds` 均从配置加载，不在场景逻辑中重复定义。

完整帧字段和同步规则见 [`docs/data_chain.md`](docs/data_chain.md)。

YOLO 的设备配置在 [`config/perception.yaml`](config/perception.yaml)。当前
`device: auto` 会自动使用可用 CUDA，否则回退 CPU；每帧感知结果会记录实际
设备。20 Hz（50 ms）仍是设计基准，是否采用该频率要以端到端实测耗时为准。

## 当前联调阶段

当前只要求打通传感器数据链。规划和控制后续接入，背景交通数量也后续扩展。建议先启用 `front_rgb`，完成 100 m 直行的帧号、时间戳、RGB 转换、日志和丢帧检查，再启用双目相机和标定文件。

端到端联调入口：

```bash
$PROJECT_PYTHON -m perception_runtime.run_carla_data_chain \
  --config My_scenario/configs/my_scenario.yaml \
  --distance-m 100
```

运行运行时单测：

```bash
$PROJECT_PYTHON -m unittest -v perception_runtime.test_runtime
```

运行原有车辆/行人帧接口单测：

```bash
PYTHONPATH=yolov5 $PROJECT_PYTHON -m unittest -v yolov5.test_frame_perception
```
