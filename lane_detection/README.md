# UFLD 车道感知

当前模块使用 UFLD v1 的 CULane、ResNet18 权重进行单张图片车道候选点检测。模型最多提供四个模型槽位；槽位不是跨帧跟踪 ID，也不代表一定存在四条有效车道。

## 正式单图入口

从任意工作目录运行：

```bash
cd /tmp
export PROJECT_PYTHON=/data/hdt_workspace/my_env/carla0915/bin/python3.7
$PROJECT_PYTHON \
  /data/hdt_workspace/carla-perception/lane_detection/lane_detection.py \
  --source /data/hdt_workspace/carla-perception/test_data/lane_images/001.jpg \
  --ufld-dir /data/hdt_workspace/carla-perception/ufld \
  --weights /data/hdt_workspace/carla-perception/lane_detection/models/culane_18.pth \
  --device cpu \
  --trusted-legacy-checkpoint \
  --output-root /data/hdt_workspace/carla-perception/outputs
```

`--device 0` 只在本机 CUDA 设备可用时使用；不可用时程序会明确失败，不静默切换 CPU。`--trusted-legacy-checkpoint` 仅用于已经核验来源的旧式 UFLD checkpoint，不会修改全局 PyTorch 加载设置。

## 可复用接口

```python
import cv2
from lane_detection.ufld_detector import UFLDDetector

detector = UFLDDetector(
    ufld_dir="/data/hdt_workspace/carla-perception/ufld",
    weights="/data/hdt_workspace/carla-perception/lane_detection/models/culane_18.pth",
    device="cpu",
    trusted_legacy_checkpoint=True,
)
bgr = cv2.imread("image.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
result = detector.detect(rgb, metadata={
    "episode_id": "episode-01",
    "camera_id": "front",
    "frame_id": 12,
    "timestamp_ns": 1234567890,
    "frame_id_source": "sensor",
    "timestamp_source": "sensor",
})
```

模型只在 `UFLDDetector` 初始化时加载一次。输入必须是未修改的 `uint8`、`H x W x 3`、RGB 数组。

## 输出与状态

单图输出继续保存到：

```text
outputs/lane_detection/YYYYMMDD_HHMMSS_<输入stem>/
├── run.json
└── samples/<输入stem>/
    ├── 00_input/original.png
    ├── 01_detection/lane_points.jpg
    └── 02_result/
        ├── result.jpg
        └── result.json
```

`result.json` 的 `lanes` 保存原图像素坐标候选点、采样行索引、逐行存在分数、模型槽位和有效状态。`status=ok` 且 `lanes_valid=true` 表示模型推理及输出解码有效，即使 `valid_lane_count=0` 也不表示推理错误。输入、模型加载或推理失败时为 `status=error`、`lanes_valid=false`，并在 `errors` 或 `run.json.error_summary` 中记录原因。

`run.json` 分开记录模型初始化、预处理、网络前向、传输/后处理和可视化保存耗时。GPU 前向计时会同步 CUDA；单次运行耗时不等于稳定 FPS 或完整 20 Hz 链路性能。

## 当前能力边界

- 当前输出是车道候选槽位和候选点，不确认本车左右边界。
- 项目中未发现可靠的相机内参、相机相对车辆外参、车辆参考点、坐标系约定或道路平面参数，因此 `ego_boundary_role` 保持 `unknown`，左右边界 ID 为 `null`。
- 当前未实现米制横向距离：`metric_distances_valid=false`，左右距离为 `null`。后续需要相机标定、安装姿态、车辆参考点、道路平面/车道几何以及明确的距离测量位置和符号约定。
- 不将像素距离当作米，不把图像中心或固定模型槽位当作本车参考。
- 不包含视频读取、CARLA 客户端、跨帧跟踪、SGBM、速度估计或通行判断。

在获得上述标定和道路几何参数前，不具备可靠实现本车边界和米制距离的条件；也不应据此开始需要深度/双目的 SGBM 定位原型验收。
