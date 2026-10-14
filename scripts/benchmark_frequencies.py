"""Run the synchronized CARLA chain at the requested sensor frequencies."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_one(args, hz):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "yolov5")
    command = [
        sys.executable, "-m", "perception_runtime.run_carla_data_chain",
        "--config", str(Path(args.config).resolve()),
        "--frequency-hz", str(hz),
        "--max-frames", str(args.max_frames),
        "--throttle", str(args.throttle),
        "--evaluation-profile", "mixed",
    ]
    if args.remote_inference_port is None:
        command.append("--save-yolo-debug")
    else:
        command.extend(["--remote-inference-port", str(args.remote_inference_port)])
    completed = subprocess.run(command, cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, check=True)
    output_dir = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("data chain complete: "):
            output_dir = Path(line.split(": ", 1)[1].strip())
            break
    if output_dir is None:
        raise RuntimeError("runner did not report output directory\n" + completed.stdout)
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    acceptance_command = [sys.executable, "-m", "scripts.validate_yolo_episode", str(output_dir)]
    if args.remote_inference_port is not None:
        acceptance_command.append("--allow-no-debug")
    validation = subprocess.run(acceptance_command, cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, check=True)
    acceptance = json.loads(validation.stdout)
    bridge = summary["sensor_bridge"]
    bridge_start = summary.get("sensor_bridge_start", {})
    received = (int(bridge.get("frames_received", {}).get("front_rgb", 0)) -
                int(bridge_start.get("frames_received", {}).get("front_rgb", 0)))
    dropped = (int(bridge.get("frames_dropped", {}).get("front_rgb", 0)) -
               int(bridge_start.get("frames_dropped", {}).get("front_rgb", 0)))
    stale = (int(bridge.get("frames_discarded_stale", {}).get("front_rgb", 0)) -
             int(bridge_start.get("frames_discarded_stale", {}).get("front_rgb", 0)))
    processed = int(summary["frames_processed"])
    return {
        "frequency_hz": float(hz),
        "episode": str(output_dir),
        "frames_processed": processed,
        "achieved_wall_fps": summary.get("achieved_wall_fps"),
        "real_time_factor": summary.get("real_time_factor"),
        "frames_received": received,
        "frames_dropped": dropped,
        "frames_discarded_stale": stale,
        "frames_unprocessed_after_run": max(0, received - processed),
        "frames_unaccounted": max(0, received - processed - dropped - stale),
        "frame_alignment": acceptance["frame_alignment"],
        "truth_isolation": acceptance["truth_isolation"],
        "inference_latency_ms": acceptance["latency_ms"],
        "total_latency_ms": {
            "mean": summary["total_latency_ms"]["mean"],
            "p95": summary["total_latency_ms"]["p95"],
            "max": summary["total_latency_ms"]["max"],
        },
        "metrics": acceptance["metrics"],
        "status": acceptance["status"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="My_scenario/configs/my_scenario_test.yaml")
    parser.add_argument("--frequencies", nargs="+", type=float,
                        default=[5.0, 8.0, 10.0, 15.0, 20.0])
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument("--remote-inference-port", type=int)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "phase3_frequency_baseline.json")
    args = parser.parse_args()
    results = []
    for hz in args.frequencies:
        print("running {:.1f} Hz".format(hz), flush=True)
        results.append(run_one(args, hz))
    report = {
        "config": str(Path(args.config).resolve()),
        "frequencies_hz": list(args.frequencies),
        "runs": results,
        "selection_rule": "highest frequency with aligned frames, no runtime sensor drops, total P95 below its frame period and achieved wall FPS at or above requested Hz",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
