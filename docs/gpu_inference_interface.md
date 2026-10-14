# 独立 GPU 推理接口

CARLA/Python 3.7 侧通过回环 TCP 向 Python 3.11 GPU 服务发送长度前缀 JSON 头和原始 RGB 字节。头部字段严格限定为 `episode_id`、`camera_id`、`frame_id`、`timestamp_ns`、`image_width_px`、`image_height_px`、`image_rgb_bytes`。服务拒绝其他字段，真值、actor 和场景状态无法进入推理请求。响应只返回检测框、耗时、设备、精度和相同帧元数据，`source` 固定为 `camera_model`。

启动真实服务：

```bash
PYTHONPATH="$PWD:$PWD/yolov5" /data/hdt_workspace/my_env/drive-jepa-navsim/bin/python \
  -m perception_runtime.gpu_inference_server --port 46099
```

CARLA 端运行时增加 `--remote-inference-port 46099`。服务读取 `config/perception.yaml` 中的 YOLO 权重、阈值、图像尺寸和 FP16 设置。端口可按需更改。

协议回归：

```bash
source scripts/project_env.sh
$PROJECT_PYTHON scripts/verify_gpu_inference_interface.py
```

真实 CUDA 结果和墙钟频率见 `docs/gpu_inference_status.md`。
