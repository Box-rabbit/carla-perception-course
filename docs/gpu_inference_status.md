# GPU 推理状态（2026-10-14）

CARLA 主进程继续使用 Python 3.7.16 和 CPU PyTorch 1.13.1。该 PyTorch wheel 不支持 RTX 5060 的 `sm_120`。独立推理进程使用 `/data/hdt_workspace/my_env/drive-jepa-navsim/bin/python`（Python 3.11.15、PyTorch 2.11.0+cu128），已实测识别 RTX 5060 并运行 `cuda:0 / fp16` YOLOv5s。上游 YOLOv5 文件未改动。

真实 CARLA 100 帧远端推理验收见 `docs/phase4_gpu_acceptance.json`：接口、真值隔离、帧对齐均通过；推理 P95 9.93 ms，感知处理 P95 13.61 ms。该次运行有 1 帧 262 ms 的峰值。随后二进制 RGB 传输的 100 帧测试见 `docs/phase3_gpu_20hz_sustained.json`：推理 P95 9.67 ms，感知处理 P95 13.06 ms，最长 32.81 ms，未发生运行期传感器丢帧。

**频率结论**：20 Hz 是仿真固定步长；100 帧运行的墙钟吞吐为 11.03 帧/秒，尚未达到实时 20 Hz。5/8/10/15/20 Hz 的 50 帧 GPU 基线见 `docs/phase3_gpu_frequency_baseline.json`，当前机器在各档均约 10.8 帧/秒。建议当前实时演示使用 8 Hz；10 Hz 已达到平均墙钟吞吐，但余量较小。下一优化目标是 CARLA 渲染/同步链路。
