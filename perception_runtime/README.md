# CARLA 感知运行时

本模块是场景到感知的在线桥接层，不实现规划和控制。

## 数据链入口

从仓库根目录运行：

```bash
source scripts/project_env.sh

$PROJECT_PYTHON -m perception_runtime.run_carla_data_chain \
  --config My_scenario/configs/my_scenario.yaml \
  --distance-m 100
```

运行器读取 `config/system.yaml` 和 `config/sensors.yaml`，在同步 CARLA 世界中安装启用的相机，按 `world.tick()` 取出相同帧号的图像，交给 `PerceptionRuntime`，并写出：

- `outputs/data_chain/<episode_id>/perception.jsonl`：仅相机输入和模型结果；
- `outputs/data_chain/<episode_id>/truth.jsonl`：CARLA snapshot、actor 状态和场景观测，仅供评测。

场景真值不作为处理器参数传递。相机安装参数在 `config/sensors.yaml`，全局 `sensor_tick_seconds=0.05` 在 `config/system.yaml`。

## 帧同步

`CarlaSensorBridge` 将 CARLA 的 BGRA 回调转为 RGB，并按 `world.tick()` 返回的 CARLA `frame_id` 配对所有启用相机。缺帧、超前帧、尺寸错误和时间戳不一致会被拒绝或计入丢帧统计。处理器输出使用 `source: camera_model`；真值日志使用独立的 `source: ground_truth`。

当前空处理器注册表用于先验证传感器数据链。规划控制、YOLO、UFLD、SGBM 和跟踪器可以在数据链稳定后作为处理器接入。

## YOLOv5 接入验收

YOLO 配置在 `config/perception.yaml`，不属于场景配置。在线运行时只把
`front_rgb` 的 `SensorFrame.image_rgb` 传给模型；真值仍只写入独立的
`truth.jsonl`。

`device: auto` 会在 CUDA 可用时选择 `cuda:0`，否则使用 CPU；`half: true`
只在 CUDA 上启用 FP16。若显式配置 `cuda:0` 而驱动或 CUDA 不可用，程序会
直接报错，避免把 GPU 配置静默降级成 CPU。每帧结果记录
`device_requested`、`device_actual` 和 `precision`，便于确认实际运行设备。
当前 Python 3.7 环境中的 PyTorch 1.13 CUDA wheel 不支持 RTX 5060 的
`sm_120`，所以本机 `auto` 会安全回退 CPU；详见
[`docs/gpu_inference_status.md`](../docs/gpu_inference_status.md)。

离线单图验收（生成 JSON、原图、画框图和耗时报告）：

```bash
source scripts/project_env.sh
export PYTHONPATH="$PWD/yolov5:$PWD"
$PROJECT_PYTHON scripts/validate_yolo_integration.py
```

在线 20 帧验收并保存原图/画框图：

```bash
$PROJECT_PYTHON -m perception_runtime.run_carla_data_chain \
  --config My_scenario/configs/my_scenario.yaml \
  --distance-m 1000 --max-frames 20 --save-yolo-debug
```

不修改正式场景即可运行临时感知夹具。车辆基线：

```bash
$PROJECT_PYTHON -m perception_runtime.run_carla_data_chain \
  --config My_scenario/configs/my_scenario.yaml \
  --distance-m 1000 --max-frames 100 --throttle 0 \
  --evaluation-profile vehicle --save-yolo-debug
```

车辆、远车和行人混合夹具：

```bash
$PROJECT_PYTHON -m perception_runtime.run_carla_data_chain \
  --config My_scenario/configs/my_scenario.yaml \
  --distance-m 1000 --max-frames 100 --throttle 0 \
  --evaluation-profile mixed --save-yolo-debug
```

输出目录下的 `yolo_debug/raw` 与 `yolo_debug/annotated` 用于人工检查，
`perception.jsonl` 中的 `inference_latency_ms` 用于耗时检查。

在线结束后，用独立真值日志评估 2D IoU（真值不会进入推理）：

```bash
$PROJECT_PYTHON scripts/evaluate_yolo_truth.py outputs/data_chain/<episode_id>
```

一键验收（帧对齐、真值隔离、状态、图片数量、耗时和按类别指标）：

```bash
$PROJECT_PYTHON scripts/validate_yolo_episode.py \
  outputs/data_chain/<episode_id>
```

验收结果会写入 `yolo_acceptance.json`。`status: pass` 表示全部通过；
`pass_with_latency_over_budget` 表示功能和数据链通过，但有帧超过 50 ms，
需要记录性能问题后再决定是否换模型或使用 GPU。`yolo_truth_metrics.json`
包含总体及 `vehicle`、`pedestrian` 分类别的 Precision、Recall、F1 和 IoU。

若相机视野内没有车辆/行人，评估会报告 `evaluation_status: no_visible_truth`，
这表示没有可评估目标，不表示模型精度为零。
