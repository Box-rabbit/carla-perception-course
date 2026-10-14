# 免训练灯色分类器：Autoware MobileNetV2 v2

## 正式单图联合入口

联合程序位于 `traffic_light_pipeline.py`，从原图完成 YOLOv5 class 9 信号灯检测、原图裁剪、预训练灯色分类和结果保存。它只支持单张图片，不负责视频、CARLA、跟踪或通行判断。

```bash
cd /data/hdt_workspace/carla-perception
export PROJECT_PYTHON=/data/hdt_workspace/my_env/carla0915/bin/python3.7
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $PROJECT_PYTHON \
traffic_light_classifier/traffic_light_pipeline.py \
  --source /data/hdt_workspace/carla-perception/test_data/images/image.png \
  --yolo-dir /data/hdt_workspace/carla-perception/yolov5 \
  --weights /data/hdt_workspace/carla-perception/yolov5/yolov5s.pt \
  --model-dir /data/hdt_workspace/carla-perception/traffic_light_classifier/models \
  --device 0 \
  --output-root /data/hdt_workspace/carla-perception/outputs
```

结果保存到 `outputs/traffic_light/YYYYMMDD_HHMMSS_<输入stem>/`，每次重名运行自动追加 `_01`、`_02`，不会覆盖旧结果。详细输出结构以项目冻结的“感知算法输出目录基线 v1”为准。

## 本阶段范围

输入：从未画框的原图裁出的单个信号灯灯头 RGB 图像。
输出：`red / yellow / green / unknown`，原始模型标签、模型置信度和分类原因。
这是独立 ROI 分类器，不修改已有 YOLOv5s 车辆/行人检测接口，不训练，不安装 ROS、Autoware、CUDA 或新的深度学习环境。
使用现有 NumPy、OpenCV DNN 在 CPU 推理；显卡继续留给 YOLO。模型只在启动时加载一次。一个实例请串行调用。

## 1. Linux 放置代码

把压缩包传到 `/data/hdt_workspace/carla-perception/`，解压后应有：

```text
/data/hdt_workspace/carla-perception/traffic_light_classifier/
  model_assets.py
  traffic_light_classifier.py
  demo_classifier.py
  test_classifier.py
  使用说明.md
```

```bash
cd /data/hdt_workspace/carla-perception
unzip -n traffic_light_classifier.zip
export PROJECT_PYTHON=/data/hdt_workspace/my_env/carla0915/bin/python3.7
cd /data/hdt_workspace/carla-perception/traffic_light_classifier
$PROJECT_PYTHON -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

统一 Python 3.7 环境已安装原生 OpenCV 4.6.0.66 CPU 版本。运行前先执行 `source scripts/project_env.sh`，确保使用项目解释器。出现 ONNX 算子错误时保留输出，不随意重装原环境。

## 2. 下载已训练好的模型

```bash
$PROJECT_PYTHON model_assets.py
```

程序从官方部署使用的地址下载以下两个文件到 `models/`，校验长度与 SHA256；已有正确文件不会被覆盖。

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| traffic_light_classifier_mobilenetv2_batch_1.onnx | 8926508 | 455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703 |
| lamp_labels.txt | 118 | 1a5a49eeec5593963eab8d70f48b8a01bfb07e753e9688eb1510ad26e803579d |

下载超时时，可在能联网的电脑浏览器下载两个文件，传到上述 `models/` 目录，再运行 `$PROJECT_PYTHON model_assets.py` 校验：

- [官方模型](https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v2/traffic_light_classifier_mobilenetv2_batch_1.onnx)
- [官方标签](https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v2/lamp_labels.txt)

压缩包不附模型权重。官方源代码使用 Apache-2.0；尚未核实权重单独的许可条款，不将源代码许可证直接当成权重许可证。对外分发、商用前需另行核实权重授权。

## 3. 跑接口自测

```bash
$PROJECT_PYTHON -m unittest -v test_classifier
```

应显示 `Ran 13 tests` 和 `OK`。该步骤验证预处理、标签、阈值、异常输入和 ROI，不代替真实模型推理。

## 4. 准备单个灯头图片

推荐先沿用你已成功的 YOLO 检测命令，添加 `--save-crop`。只筛选信号灯的示例：

```bash
cd /data/hdt_workspace/carla-perception/yolov5
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $PROJECT_PYTHON detect.py \
  --weights yolov5s.pt \
  --source /data/hdt_workspace/carla-perception/test_data/images/image.png \
  --imgsz 640 --device 0 --classes 9 \
  --conf-thres 0.25 --iou-thres 0.45 \
  --save-crop --project ../outputs --name traffic_light_crops
```

成功检测时，裁图通常保存在：
`/data/hdt_workspace/carla-perception/outputs/traffic_light_crops/crops/traffic light/`。
若名字已存在，YOLO 自动追加数字，以终端实际路径为准。
官方裁图有少量扩边；检查每张是否只有一个完整灯头，不能把三个灯一起送入分类器。
如果先前仅用更低检测阈值才能检测成功，沿用先前的检测参数；分类置信阈值是另一回事。
不要用带检测框、类别文字的输出图进行分类。

也可以手工裁成 PNG/JPG 放进 `test_data/traffic_light_rois/`。无固定原图尺寸要求；程序自动等比缩放到 224×224。太小、模糊、曝光严重的 ROI 可能不能可靠分类，放大不能补回细节。

## 5. 分类并输出 JSON

一张裁图：

```bash
cd /data/hdt_workspace/carla-perception/traffic_light_classifier
$PROJECT_PYTHON demo_classifier.py \
  --source /data/hdt_workspace/carla-perception/test_data/traffic_light_rois/light_01.png \
  --min-confidence 0.70 \
  --output ../outputs/traffic_light_classifier/light_01.json
```

直接处理 YOLO 的全部灯头裁图：

```bash
$PROJECT_PYTHON demo_classifier.py \
  --source "/data/hdt_workspace/carla-perception/outputs/traffic_light_crops/crops/traffic light" \
  --min-confidence 0.70 \
  --output ../outputs/traffic_light_classifier/image_lights.json
```

含空格的目录必须加引号。输出文件存在时不会覆盖，复测请换新输出文件名。
如果结果为错误，终端与 JSON 标出 `status=error, prediction_valid=false`，程序返回非零退出码；`status=ok, state=unknown` 表示推理完成但不输出单一颜色，两者不同。

无需落盘裁图时，支持在原图上指定原像素坐标，右边/下边不包含在裁图内：

```bash
$PROJECT_PYTHON demo_classifier.py \
  --source /data/hdt_workspace/carla-perception/test_data/images/image.png \
  --roi 379 249 411 322 \
  --output ../outputs/traffic_light_classifier/manual_roi.json
```

最后这个框仅适用于你发来的 **814×607 原图**中间灯头。若 Linux 原文件的尺寸、裁剪或内容不同，不要照搬坐标，请用那张图自己的 YOLO 框。

## 6. 判断结果，不误解标签

```json
{
  "status": "ok",
  "prediction_valid": true,
  "state": "green",
  "model_confidence": 0.99,
  "model_debug": {
    "raw_label": "green",
    "top3": [{"label": "green", "probability": 0.99}]
  },
  "reason": "single_color_prediction"
}
```

- 置信度是模型最优原始类别概率，不是安全概率、准确率或 YOLO×分类器端到端概率。
- `unknown` 有三种推理原因：模型 unknown 类、低于阈值、同一 ROI 多种颜色组合。
- 该模型有 11 个类别，**不是仅三个类别**。`left,red` 的官方含义是“绿色左箭头 + 红色圆灯”，不是“红色左箭头”。组合类保留元素，标为 unknown，不能一概归成红色。
- **红色箭头不在该版本的独立类别中**。你的左侧红箭头手工 ROI 在本机被判为 `red`，但模型同时解释为圆灯，不能据此宣称支持红色箭头形状或通行方向。两个绿色圆灯 ROI 本机均判为 green；这是样例检查，不是精度评测。
- 暂时不能仅凭分类结果决定哪个灯控制本车、是否允许通行；灯头与车道关联还要等场景侧与后续模块。
- 先分别测试红/黄/绿圆灯及无亮灯、遮挡、模糊情况，保留原图、YOLO 框和分类 JSON。你目前的图没有黄灯，不能声称已经验证黄灯性能。
- 不使用 CARLA actor 状态替代视觉分类，不把本工具说成完成全部课程接口或验收。

## 后续内存接口

```python
from traffic_light_classifier import TrafficLightClassifier

classifier = TrafficLightClassifier()
rgb_roi = rgb_image[top:bottom, left:right]
result = classifier.classify(rgb_roi)
```

`rgb_image` 必须是 uint8 RGB；OpenCV 读取默认 BGR，需先转换。模型实例在循环外创建。
本阶段先跑通 ROI 分类，之后再把 YOLO class 9 的框接入该接口，并补齐帧号、时间戳和灯头关联。

## 固定来源与预处理

- [官方部署清单，固定 0.41.0](https://github.com/autowarefoundation/autoware/blob/0.41.0/ansible/roles/artifacts/tasks/main.yaml)
- [官方预处理，固定 0.41.0](https://github.com/autowarefoundation/autoware_universe/blob/0.41.0/perception/autoware_tensorrt_classifier/src/tensorrt_classifier.cpp)
- [官方标签解码，固定 0.41.0](https://github.com/autowarefoundation/autoware_universe/blob/0.41.0/perception/autoware_traffic_light_classifier/src/classifier/cnn_classifier.cpp)
- [分类模块说明](https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_traffic_light_classifier/)

RGB、NCHW、224×224，等比缩放、左上对齐、仅右/下补原始黑色；均值 `[123.675,116.28,103.53]`、标准差 `[58.395,57.12,57.375]`。输出已带 Softmax，不再重复 Softmax。
仅极端长宽比导致缩放边长小于 1 时，边长夹到 1 以避免 OpenCV 零尺寸错误。
训练域与 CARLA 不同；免训练先建立可运行基线，不承诺满足你的场景精度。
