"""新文件：加载其它同学写的 agent，不控制自车。

用途：把场景交给同学的 agent 跑，验证 agent 能不能完成场景任务。
自车由同学的 agent 控制，本文件只负责场景。
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from common.carla import add_carla_pythonapi
from common.config import DEFAULT_SCENARIO_CONFIG, load_runtime_config, validate_python_runtime

add_carla_pythonapi()

import carla
from scenarios_impl.my_scenario import build_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_SCENARIO_CONFIG))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    
    # 从环境变量加载同学的 agent
    module_name = os.environ.get("MY_AGENT_MODULE", "").strip()
    if not module_name:
        raise RuntimeError(
            "MY_AGENT_MODULE is not set. "
            "Set it to your teammate's agent module."
        )
    
    module = importlib.import_module(module_name)
    factory = getattr(module, "build_agent", None)
    if factory is None:
        raise RuntimeError(f"{module_name} must expose build_agent(config_path)")
    agent = factory(args.config)
    print(f"[AGENT] 已加载 agent: {module_name}")
    
    config = load_runtime_config(scenario_path=args.config)
    validate_python_runtime(config["_system"])
    # 连接 CARLA
    client = carla.Client(
        args.host or config["_system"]["carla"]["host"],
        int(args.port or config["_system"]["carla"]["port"]),
    )
    client.set_timeout(20.0)
    world = client.get_world()
    print("[AGENT] 已连接 CARLA")
    
    # 加载场景
    scenario = build_scenario()
    print("[AGENT] 场景已加载")
    
    # TODO: 生成 ego、生成 NPC、运行循环，每帧调用 agent.run_step()
    
    if hasattr(agent, "destroy"):
        agent.destroy()
    print("[AGENT] 场景运行完成")


if __name__ == "__main__":
    main()
