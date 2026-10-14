# 阶段一至四实施状态

更新时间：2026-10-14

## 阶段一：真实 CARLA 数据链

- 状态：完成
- 配置：`My_scenario/configs/my_scenario_test.yaml`
- 结果：CARLA 0.9.15、Town05、front_rgb、同步帧、真值日志和感知日志全部打通。
- 验收输出：`outputs/data_chain/town05_data_chain_20261014_012155_986610`
- 接口/隔离/对齐：通过。
- 20 帧混合目标评估：Precision 0.952、Recall 1.000、F1 0.976、平均 IoU 0.660。
- 延迟：CPU YOLO 均值 129.20 ms，P95 140.99 ms，因此状态为 `pass_with_latency_over_budget`。

## 阶段二：正式场景检测评估

- 状态：完成
- 配置：`My_scenario/configs/my_scenario.yaml`
- 100 帧真值可见率：100/100。
- 综合结果：Precision 0.843、Recall 0.967、F1 0.901、平均 IoU 0.662。
- 行人：Precision/Recall/F1 均为 1.000。
- 车辆：Precision 0.779、Recall 0.950、F1 0.856。
- 原图和画框图：各 100 张；JSON 接口、真值隔离、帧对齐通过。
- 详细验收：`docs/phase2_formal_acceptance.json`。

## 阶段三：频率基线

- 状态：完成初版基线
- 频率：5、8、10、15、20 Hz 均已实测。
- 所有频率均保持 frame_id/timestamp 对齐；10 帧基线中每次运行均处理 10 帧、无传感器丢帧，仅有 1 个 warmup 后残留帧被丢弃。CPU 推理均值约 122–125 ms，整帧均值约 192–199 ms，50 ms 预算全部超限。
- GPU 独立进程在 5/8/10/15/20 Hz 的 50 帧基线中，感知处理 P95 约 10.5–15.1 ms，但实际墙钟吞吐均约 10.8 帧/秒。20 Hz 的 100 帧持续测试实际为 11.03 帧/秒。
- 当前建议实时演示使用 8 Hz；10 Hz 平均可达到，但余量较小。20 Hz 仍为仿真频率，需继续优化 CARLA 渲染/同步链路。
- 详细数据：`docs/phase3_frequency_baseline.json`（CPU）、`docs/phase3_gpu_frequency_baseline.json`（GPU）、`docs/phase3_gpu_20hz_sustained.json`（GPU 100 帧）。
- 频率结果摘要：`docs/phase3_gpu_frequency_baseline.md`。

## 阶段四：独立 GPU 推理接口

- 状态：真实 GPU 验证完成
- 新增 `perception_runtime/gpu_inference_server.py` 和 `perception_runtime/remote_inference.py`。
- CARLA Python 3.7 进程只发送 RGB 图像、camera_id、frame_id、timestamp_ns 等相机数据；不传 actor、场景状态或真值。
- 返回结果固定为 `source: camera_model`。
- mock server 3 帧协议回归：通过。真实 RTX 5060 `cuda:0 / fp16` 的 CARLA 100 帧验收：通过，F1 0.906、推理 P95 9.93 ms。
- 原始 RGB 二进制传输的 100 帧测试：感知处理 P95 13.06 ms、无运行期丢帧。
- 详细验收：`docs/phase4_gpu_acceptance.json`；接口：`docs/gpu_inference_interface.md`。

## 已知后续工作

1. 优化 CARLA 渲染/同步链路，将实际墙钟吞吐从约 11 帧/秒提高到 20 帧/秒，并做长时间稳定性测试。
2. 继续扩展场景车辆/行人数量和规控模块；这些不阻塞当前感知数据链。
