"""Verify the independent camera-only inference wire interface."""

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def send(stream, payload, image):
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    stream.sendall(struct.pack("!I", len(encoded)) + encoded + image.tobytes())


def receive(stream):
    header = stream.recv(4)
    if len(header) != 4:
        raise RuntimeError("server closed before response")
    size = struct.unpack("!I", header)[0]
    data = bytearray()
    while len(data) < size:
        data.extend(stream.recv(size - len(data)))
    return json.loads(bytes(data).decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gpu_interface_") as temp:
        ready = Path(temp) / "ready.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "yolov5")
        command = [sys.executable, "-m", "perception_runtime.gpu_inference_server",
                   "--mock", "--port", "0", "--ready-file", str(ready)]
        process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(0.05)
            if not ready.exists():
                output = process.stdout.read().decode("utf-8", "replace") if process.stdout else ""
                raise RuntimeError("server did not become ready: " + output)
            address = json.loads(ready.read_text(encoding="utf-8"))
            with socket.create_connection((address["host"], address["port"]), timeout=5.0) as stream:
                responses = []
                for frame_id in range(args.frames):
                    image = np.zeros((8, 12, 3), dtype=np.uint8)
                    request = {
                        "episode_id": "gpu_interface_check",
                        "camera_id": "front_rgb",
                        "frame_id": frame_id,
                        "timestamp_ns": frame_id * 50000000,
                        "image_width_px": 12,
                        "image_height_px": 8,
                        "image_rgb_bytes": int(image.nbytes),
                    }
                    send(stream, request, image)
                    responses.append(receive(stream))
            assert [item["frame_id"] for item in responses] == list(range(args.frames))
            assert all(item["source"] == "camera_model" for item in responses)
            forbidden = ("truth", "ground_truth", "actor_id", "scenario_observation")
            assert all(not any(key in item for key in forbidden) for item in responses)
            report = {"status": "pass", "frames": len(responses),
                      "source": "camera_model", "truth_isolation": "pass",
                      "frame_alignment": "pass", "device": "mock"}
            print(json.dumps(report, indent=2))
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
