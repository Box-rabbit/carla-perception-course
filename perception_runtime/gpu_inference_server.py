"""Standalone camera-only inference worker for the Python 3.7 CARLA process.

The wire protocol carries episode/camera/frame/timestamp and an RGB image only.
It never carries CARLA actors, scenario state, or truth records.
"""

import argparse
import io
import json
import socket
import struct
import time

import numpy as np


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.recv(size - len(data))
        if not chunk:
            raise EOFError("inference client disconnected")
        data.extend(chunk)
    return bytes(data)


def _recv(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if size <= 0 or size > 1048576:
        raise ValueError("invalid inference request header size")
    request = json.loads(_read_exact(stream, size).decode("utf-8"))
    allowed = {
        "episode_id", "camera_id", "frame_id", "timestamp_ns",
        "image_height_px", "image_width_px", "image_rgb_bytes",
    }
    if set(request) != allowed:
        raise ValueError("inference request contains missing or forbidden fields")
    if request["camera_id"] != "front_rgb":
        raise ValueError("unsupported camera_id")
    image_size = int(request["image_rgb_bytes"])
    expected = int(request["image_height_px"]) * int(request["image_width_px"]) * 3
    if image_size != expected or image_size <= 0 or image_size > 100000000:
        raise ValueError("invalid RGB image byte count")
    request["image_rgb"] = _read_exact(stream, image_size)
    return request


def _send(stream, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    stream.sendall(struct.pack("!I", len(encoded)) + encoded)


def _infer(request, detector):
    image = np.frombuffer(request["image_rgb"], dtype=np.uint8)
    image = image.reshape((int(request["image_height_px"]), int(request["image_width_px"]), 3))
    started = time.perf_counter()
    detections = detector.detect(image) if detector is not None else []
    latency = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": "gpu-inference-1.0",
        "episode_id": request["episode_id"],
        "camera_id": request["camera_id"],
        "frame_id": int(request["frame_id"]),
        "timestamp_ns": int(request["timestamp_ns"]),
        "source": "camera_model",
        "detections": detections,
        "inference_latency_ms": latency,
        "device": "mock" if detector is None else str(getattr(detector, "device", "cuda")),
        "precision": ("fp16" if detector is not None and detector.use_half else "fp32"),
    }


def serve(host, port, detector=None, ready_file=None):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(4)
    actual_port = server.getsockname()[1]
    if ready_file:
        with open(ready_file, "w", encoding="utf-8") as stream:
            json.dump({"host": host, "port": actual_port}, stream)
    print("GPU inference server listening on {}:{}".format(host, actual_port), flush=True)
    try:
        while True:
            connection, _address = server.accept()
            with connection:
                while True:
                    try:
                        request = _recv(connection)
                    except EOFError:
                        break
                    _send(connection, _infer(request, detector))
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file")
    parser.add_argument("--mock", action="store_true",
                        help="run protocol validation without loading a model")
    args = parser.parse_args()
    detector = None
    if not args.mock:
        from common.config import load_perception_config
        from yolov5.perception_detector import YoloV5Detector
        settings = load_perception_config()["yolov5"]
        detector = YoloV5Detector(
            weights=settings["weights"],
            image_size=int(settings.get("image_size", 640)),
            device="auto",
            half=bool(settings.get("half", False)),
            confidence_threshold=float(settings.get("confidence_threshold", 0.25)),
            iou_threshold=float(settings.get("iou_threshold", 0.45)),
            max_detections=int(settings.get("max_detections", 300)),
        )
    serve(args.host, args.port, detector, args.ready_file)


if __name__ == "__main__":
    main()
