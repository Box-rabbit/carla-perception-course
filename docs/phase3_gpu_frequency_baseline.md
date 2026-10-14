# GPU 频率基线摘要

测试环境：CARLA 0.9.15 / Town05，Python 3.7 CARLA 进程，独立 Python 3.11.15 + PyTorch 2.11.0+cu128 推理进程，RTX 5060 Laptop GPU，YOLOv5s FP16。每档 50 帧，混合车辆/行人固定目标；所有频率均无运行期传感器丢帧、frame_id/timestamp 对齐通过。

| 配置仿真频率 | 实际墙钟吞吐 | 感知处理 P95 | 实际/配置实时因子 |
|---:|---:|---:|---:|
| 5 Hz | 10.93 FPS | 10.53 ms | 2.19 |
| 8 Hz | 10.77 FPS | 13.37 ms | 1.35 |
| 10 Hz | 10.83 FPS | 12.93 ms | 1.08 |
| 15 Hz | 10.90 FPS | 15.06 ms | 0.73 |
| 20 Hz | 10.75 FPS | 13.55 ms | 0.54 |

感知计算本身很快，墙钟 FPS 却几乎不随仿真频率变化，说明当前限制主要来自 CARLA 渲染和同步推进。20 Hz 数据步长没有达到 20 FPS 实时吞吐。短测建议用 8 Hz 留余量；10 Hz 是接近实时的平均吞吐，尚未留足波动余量。

完整逐次数据：`phase3_gpu_frequency_baseline.json`。20 Hz 100 帧持续测量：`phase3_gpu_20hz_sustained.json`。后者实测 11.03 FPS，推理 P95 9.67 ms、感知处理 P95 13.06 ms。
