# 项目配置入口

项目自有配置集中在这里：

- `system.yaml`：CARLA 连接、同步时钟、数据契约、输出和标定根目录。
- `system.yaml`：同时声明项目统一使用 Python 3.7，以匹配 CARLA 0.9.15 PythonAPI。

统一环境解释器：`/data/hdt_workspace/my_env/carla0915/bin/python3.7`。仓库根目录的 `requirements-py37.txt` 是感知模块依赖清单。
- `sensors.yaml`：逻辑 `camera_id` 和每个相机的差异项；公共 blueprint、分辨率、FOV、安装基准位姿在 `defaults` 中定义。所有传感器共用的 `sensor_tick_seconds=0.05` 只在 `system.yaml` 定义。
- `../My_scenario/configs/my_scenario.yaml`：路线、ego 初始位置、触发点和场景专属参数。

加载入口是 `common.config.load_runtime_config()`（`perception_runtime.config` 仍提供兼容导入）。场景配置不再定义全局仿真时钟，传感器配置也不再从场景参数推导采样率。

`calibration_status: pending` 表示相机安装/标定尚未交付；它不影响先做 RGB 数据链，但在启用双目深度前必须替换为已验证标定文件。
