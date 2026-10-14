# SGBM 双目定位：先做标定图像对到深度，不接目标 3D 框

## 当前阶段

SGBM 是立体匹配算法，不是需要下载预训练权重的神经网络。
本入口使用现有 OpenCV CPU 后端，不需要训练、CUDA、新环境或 opencv-contrib。
执行顺序：左右原图 → 立体校正 → 左右视差 → 左右一致性过滤 → Q 三维重投影 → 深度及有效掩码。
它不负责车辆/行人检测，不输出目标中心、3D 包围框、速度或跟踪 ID。

## 1. 检查环境

在 Linux 普通终端运行：

```bash
export PROJECT_PYTHON=/data/hdt_workspace/my_env/carla0915/bin/python3.7
$PROJECT_PYTHON -c "import cv2,numpy; print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__); print('SGBM:',hasattr(cv2,'StereoSGBM_create'))"
```

SGBM 应为 True，不先升级环境。

## 2. 放置入口与素材

把交付 ZIP 放到 `/data/hdt_workspace/carla-perception/`，执行：

```bash
cd /data/hdt_workspace/carla-perception
unzip -n stereo_sgbm.zip
mkdir -p test_data/stereo_pairs/001
```

得到：

```text
carla-perception/
├── stereo_sgbm/
│   ├── stereo_localization.py
│   ├── calibration_template.json
│   └── 使用说明.md
└── test_data/stereo_pairs/001/
    ├── left.png
    ├── right.png
    └── calibration.json
```

代码目录不是结果目录，不把代码或模型放到 outputs/stereo_3d/。
输出目录创建和保存直接由 stereo_localization.py 管理，没有独立路径脚本。

## 3. 准备真实左右图

必须是同一双目装置、同步时刻、同一分辨率的左右原始图；图片尺寸必须与标定尺寸完全一致。
不能将一张图片复制成两张；不能随意找两张相似照片；不能让车辆移动拍两张图后假装已同步标定双目。
程序无法根据两张图证明同步，run.json 明确记录 synchronization_verified_by_program=false，需要采集侧保证。

CARLA 侧需要记录相同 frame 和采集 timestamp 的左右 RGB 帧，提供各自 camera_id。传感器 BGRA 原始数据转换正确后保存，不能把 RGB 当作 BGR 写入。
数据集侧需使用官方原始图及对应的真实标定。如果数据已经校正或官方只有 P/Q，需按它的几何格式适配，不能直接塞进这个原始双目模板；不重复进行不匹配的校正。
本入口支持水平排列、右相机位于左相机右侧、正视差的 OpenCV pinhole 双目，不支持鱼眼/垂直双目。

## 4. 获得标定，不猜参数

将 calibration_template.json 复制到素材目录并填真实参数：

```bash
cp -n stereo_sgbm/calibration_template.json test_data/stereo_pairs/001/calibration.json
```

模板中的 null 必须由实际标定/仿真传感器几何替换，不能仅为了运行填例子数字。模板有 null 时会故意拒绝运行。

字段：

- image_size_px：[宽, 高]。
- K_left、K_right：各自 3×3 像素内参。
- D_left、D_right：OpenCV 针孔模型畸变参数，5 参数顺序 k1,k2,p1,p2,k3；也接受官方 OpenCV 其他长度格式。只有确认无畸变时才填零。
- R_left_to_right：左光学坐标到右光学坐标的旋转。
- T_left_to_right_m：同一映射中的平移，单位米，**不是简单复制右相机的安装位置**。
- 约定：X_right = R_left_to_right @ X_left + T_left_to_right_m。
- source_description：参数来自哪个标定文件、CARLA 传感器设置或数据集。

对于平行、同向、水平双目，如果右相机在左相机右侧 B 米，则 R=I、T=[-B,0,0]。负号由上述坐标变换约定决定。真实姿态不平行时不能直接填单位旋转。
坐标轴为相机光学坐标 x 向右、y 向下、z 向前；CARLA 传感器/世界坐标不能未经变换直接套用。相机到自车的外参属于之后目标/车道自车坐标转换需要的另外一组参数。

实物双目：用同步棋盘格图像，通过真实格子尺寸及 OpenCV 单目/双目标定得到参数。仅知道相机型号不能替代标定。
CARLA：从实际分辨率、FOV、畸变设置和传感器外参推导，不能把视场角当作像素焦距。需要采集侧明确光学轴与仿真坐标转换。

## 5. 检查标定结构与双目方向

```bash
$PROJECT_PYTHON \
  /data/hdt_workspace/carla-perception/stereo_sgbm/stereo_localization.py \
  --calibration /data/hdt_workspace/carla-perception/test_data/stereo_pairs/001/calibration.json \
  --check-calibration
```

通过意味着参数结构、旋转合法性、非零基线和入口适用的双目方向通过检查，不等于标定精度已验证。

## 6. 第一次运行

```bash
$PROJECT_PYTHON \
  /data/hdt_workspace/carla-perception/stereo_sgbm/stereo_localization.py \
  --left /data/hdt_workspace/carla-perception/test_data/stereo_pairs/001/left.png \
  --right /data/hdt_workspace/carla-perception/test_data/stereo_pairs/001/right.png \
  --calibration /data/hdt_workspace/carla-perception/test_data/stereo_pairs/001/calibration.json \
  --sample-id 001 \
  --num-disparities 128 \
  --block-size 5 \
  --min-depth-m 0.5 \
  --max-depth-m 80 \
  --lr-threshold-px 1.0 \
  --output-root /data/hdt_workspace/carla-perception/outputs
```

这是 CPU SGBM，不需要 --device 0、不需要 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD。
参数范围初值不是课程验收指标。num_disparities 为 16 的正倍数；匹配窗口为奇数。不能像 YOLO 一样任意把图片缩到 640 而不更新标定。
输出保留 001 前导零、同名不覆盖。双目使用显式 --sample-id 001 关联同一对，而不是让所有素材都叫 left。

## 7. 建议的双目输出结构（新增建议，尚未冻结内部层级）

```text
outputs/stereo_3d/YYYYMMDD_HHMMSS_001/
├── run.json
└── samples/001/
    ├── 00_input/
    │   ├── left.png
    │   ├── right.png
    │   └── calibration.json
    ├── 01_rectification/
    │   ├── left_rectified.png
    │   ├── right_rectified.png
    │   ├── epipolar_check.jpg
    │   └── geometry.json
    ├── 02_disparity/
    │   ├── disparity_px.npy
    │   ├── disparity_color.png
    │   └── valid_disparity_mask.png
    └── 03_result/
        ├── depth_m.npy
        ├── xyz_m.npy
        ├── depth_color.png
        ├── valid_depth_mask.png
        └── result.json
```

先核对 epipolar_check.jpg：同一物体的同一特征点应处于左右图同一横线高度附近。如果存在明显垂直错位，先检查标定、同步和左右顺序，不先通过调 SGBM 掩盖。
disparity_color.png 是视差显示，大视差通常更近；depth_color.png 是光轴深度显示，大深度更远，两者颜色趋势不同。色彩图不是实际数值，黑色是无效。
原始 SGBM 是固定点视差，入口已除以 16 存为像素单位 float32；后续读 NPY 不再除以 16。
XYZ 单位米，坐标为**校正后的左光学相机系**，不是 CARLA 世界系、自车系或未校正左相机系。geometry.json 保存左校正旋转，可用于后续转换。
Z 是光轴方向深度，不是欧氏距离；欧氏距离需要 sqrt(X²+Y²+Z²)。
NaN 用于 NPY 无效像素，必须结合掩码处理；JSON 不写 NaN，缺失测量写 null。
透明/反光区域、低纹理、遮挡、远距离可能大量无效。有效掩码是匹配筛选结果，不是已验证精度的保证。
正常处理但没有有效深度时 status=ok、has_valid_depth=false；输入/标定/保存错误是 status=error 并非零退出。

## 8. 查询一个像素的三维坐标

在同一运行命令中增加：

```bash
--probe 500 300
```

这是**校正后左图**的 x,y 像素，不要照搬原始左图坐标。请先查看校正图并选择目标上的实际点；示例坐标仅说明参数语法。result.json 的 probe 会给 valid 和 xyz_m，无效时 xyz_m 为 null。
也可加载数组：

```python
depth = np.load(depth_path, allow_pickle=False)
xyz = np.load(xyz_path, allow_pickle=False)
```

数组 shape 分别为 H×W 和 H×W×3，索引为 [y,x]。

## 9. 调参和距离验证顺序

1. 先证明标定和极线对齐，再调匹配。
2. 对校正后相机，近似 Z=f_rectified*B_rectified/d；正式入口用 Q，考虑完整校正几何。
3. 用近处需要覆盖的视差选择范围，num_disparities 向上取 16 倍数，同时保证图像足够宽。
4. block_size 可尝试 5/7/9；大窗口可能更平稳但损失边界细节，不是越大越好。
5. 不随意放宽一致性/唯一性来提高有效像素数，数量变多不等于准确。
6. 选若干已知距离、有纹理目标区域验证近中远深度，使用独立测量或独立 CARLA 评测通道对齐，不能把真值用于推理。
7. 远距视差很小，误差会放大，需用数据验证 50m 工作范围。不能仅设置 max_depth_m=50 就声称已满足。

接 YOLO 是下一阶段：检测校正后左图，或把原图检测坐标正确变换到校正图。不能直接将原图框索引进校正视差数组。框中深度中位数可做初始候选，但并非目标真实 3D 中心，且受背景、遮挡和表面深度影响。车辆 3D 框还需尺寸和朝向估计。

## 10. 尚无同步双目和标定怎么办

可以放置代码、检查依赖和准备标定模板，但实际距离定位到此为止。下一步向场景侧索取同步左右图、准确的内参/畸变、左右位姿、单位、坐标系和采集元数据。
也可选有官方标定的公开双目数据，先按数据集格式适配；不要下载数据后猜内参或基线。

本交付在统一 Python 3.7 环境的原生 OpenCV 4.6.0.66 CPU 上完成：未填模板拒绝、零/反向基线拒绝、输入校验、人工已知 16px 视差与 5m 深度重建、完整 CLI 文件输出及重读检查。人工几何测试不是实际标定，不证明真实道路精度。
