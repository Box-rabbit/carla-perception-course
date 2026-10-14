# 场景层

场景层只负责 CARLA 世界、路线、交通灯周期、触发事件和场景真值记录。它不创建感知模型，也不把真值传给感知运行时。

## 文件

```text
My_scenario/
├── configs/my_scenario.yaml       # 场景专属配置
├── configs/my_scenario_test.yaml  # 多目标测试配置
├── configs/my_scenario_v0.yaml    # v0 历史路线配置
├── routes/my_scenario.xml         # 当前路线锚点
├── routes/my_scenario_test.xml    # 多目标测试路线
├── routes/my_scenario_v0.xml      # v0 历史路线
├── scenarios_impl/my_scenario.py  # 触发和 NPC 行为
├── scenarios_impl/route_tracker.py
├── standard_srunner_scenarios/    # ScenarioRunner 标准场景 XML
├── readme_sensor.md               # 传感器接入参考
├── run_my_scenario_test.py        # 场景临时控制器
└── run_my_scenario_agent.py       # 后续 agent 接入口
```

全局时钟、CARLA 地址、传感器契约和输出路径位于仓库根目录的 `config/`，不要在本目录重复定义。

## 运行

先设置 CARLA PythonAPI 路径（也可以直接使用已安装的 `carla`）：

```bash
source scripts/project_env.sh
$PROJECT_PYTHON My_scenario/run_my_scenario_test.py \
  --config My_scenario/configs/my_scenario.yaml \
  --duration 100
```

运行入口会从 `config/system.yaml` 读取同步设置，并将世界配置为固定步长模式。当前临时控制器仅用于场景烟囱测试；规划控制模块接入前，可以使用固定油门直行或 CARLA autopilot。

## 场景和真值接口

`MyScenario.update_state()` 生成的 `state["scenario_observation"]` 是评测真值，包含激活事件、交通灯和场景 actor。它只能写入独立的 `truth.jsonl`，不能传给 `PerceptionRuntime.process()`。

感知模型只能接收 `SensorFrameBundle`，输出写入 `perception.jsonl`。`scenario_observation` 只在场景侧生成，并由联调运行器写入独立的 `truth.jsonl`。

场景状态还提供 `perception_interface` 作为评测描述，其中的
`traffic_light_ground_truth` 只属于真值通道；它不会作为模型输入传给感知运行时。
目标频率从根目录 `config/system.yaml` 的统一时钟计算，不在场景配置中重复定义。

## 新增测试资源

`my_scenario_test.yaml` 与 `my_scenario_test.xml` 用于多目标场景测试，
`my_scenario_v0.yaml` 与 `my_scenario_v0.xml` 保留旧版路线复现入口。
`standard_srunner_scenarios/` 中的 XML 可按 ScenarioRunner 需要单独调用。

场景配置不重复声明全局 `fixed_delta_seconds` 或 `sensor_tick_seconds`；这些值统一
来自根目录 `config/system.yaml`。

## 当前场景

`my_scenario.yaml` 保留当前已验证的路线触发点。背景交通规模、复杂行人任务和规划控制属于后续阶段；本阶段以传感器数据链跑通为验收目标。
