# 标定文件

这里保存感知侧完成的相机内参和外参。`config/sensors.yaml` 只保存逻辑 `camera_id`、CARLA 安装位姿以及标定文件引用。

`front_rgb.json` 是由 CARLA 相机的分辨率和水平 FOV 推导出的虚拟相机针孔内参。已通过严格的独立 CARLA 投影验证：验证报告位于 `outputs/calibration_validation/front_rgb/20261011_153514_158500/report.json`，调试图位于同目录的 `front_rgb.png`。三个不同距离和横向位置目标的最大像素中心误差为 4.63 px，阈值为 5 px。该结果验证 CARLA 虚拟相机投影，不等价于实体相机标定。

重新验证命令（需要先启动 CARLA RPC 服务）：

```bash
source scripts/project_env.sh
$PROJECT_PYTHON calibration/validate_front_rgb_projection.py --town Town05
```

验证脚本只写独立的报告和调试图，不调用感知运行时，也不写 `perception.jsonl` 或 `truth.jsonl`。

启用 `stereo_left` 和 `stereo_right` 前，必须提供 `calibration/stereo.json`，并用 `stereo_sgbm/stereo_localization.py --check-calibration` 检查。该结构校验不能代替极线和距离精度验证。
