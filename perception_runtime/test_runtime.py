import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from perception_runtime import (
    CarlaSensorBridge,
    EpisodeLogger,
    PerceptionRuntime,
    SensorFrameBundle,
    frame_from_carla_image,
)
from perception_runtime.yolo_processor import YoloV5Processor
from perception_runtime.tracker import IoUTracker


def fake_image(frame_id, timestamp=1.0, width=2, height=1):
    # CARLA stores BGRA; this encodes RGB pixels [10, 20, 30] and [40, 50, 60].
    bgra = np.array(
        [[[30, 20, 10, 255], [60, 50, 40, 255]]], dtype=np.uint8
    )
    return SimpleNamespace(
        frame=frame_id,
        timestamp=timestamp,
        width=width,
        height=height,
        raw_data=bgra.tobytes(),
    )


class FakeSensor:
    def __init__(self):
        self.callback = None
        self.stopped = False

    def listen(self, callback):
        self.callback = callback

    def emit(self, image):
        self.callback(image)

    def stop(self):
        self.stopped = True


class RuntimeTests(unittest.TestCase):
    def test_yolo_processor_contract_uses_only_front_rgb(self):
        processor = YoloV5Processor.__new__(YoloV5Processor)
        processor.camera_id = "front_rgb"

        class FakeDetector:
            def detect(self, image):
                self.shape = image.shape
                return [{
                    "class": "vehicle", "subclass": "car",
                    "bbox2d_xyxy_px": [0, 0, 1, 1], "confidence": 0.9,
                }]

        processor.detector = FakeDetector()
        processor.debug_output_dir = None
        frame = frame_from_carla_image(fake_image(3, width=2, height=1), "episode-1", "front_rgb")
        bundle = SensorFrameBundle("episode-1", 3, frame.timestamp_ns, {"front_rgb": frame})
        result = processor(bundle)
        self.assertEqual(result["camera_id"], "front_rgb")
        self.assertEqual(len(result["detections"]), 1)
        self.assertIn("inference_latency_ms", result)
        self.assertNotIn("truth", json.dumps(result))

    def test_yolo_processor_rejects_missing_front_rgb(self):
        processor = YoloV5Processor.__new__(YoloV5Processor)
        processor.camera_id = "front_rgb"
        processor.detector = object()
        processor.debug_output_dir = None
        frame = frame_from_carla_image(fake_image(4), "episode-1", "rear_rgb")
        bundle = SensorFrameBundle("episode-1", 4, frame.timestamp_ns, {"rear_rgb": frame})
        with self.assertRaises(KeyError):
            processor(bundle)

    def test_iou_tracker_keeps_id_for_moving_detection(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=1)
        first = [{"class": "vehicle", "subclass": "car",
                  "bbox2d_xyxy_px": [10, 10, 30, 30], "confidence": 0.8}]
        second = [{"class": "vehicle", "subclass": "car",
                   "bbox2d_xyxy_px": [12, 10, 32, 30], "confidence": 0.9}]
        self.assertEqual(tracker.update(first)[0]["track_id"], 1)
        self.assertEqual(tracker.update(second)[0]["track_id"], 1)
        self.assertEqual(tracker.update([])[0]["missed_frames"], 1)
        self.assertEqual(tracker.update([]), [])

    def test_bgra_conversion_and_synchronized_bundle(self):
        left = FakeSensor()
        right = FakeSensor()
        bridge = CarlaSensorBridge("episode-1", ["left", "right"], queue_size=2)
        bridge.attach_sensor("left", left)
        bridge.attach_sensor("right", right)
        left.emit(fake_image(7, timestamp=0.35))
        right.emit(fake_image(7, timestamp=0.35))
        bundle = bridge.get_bundle(7)
        self.assertIsInstance(bundle, SensorFrameBundle)
        np.testing.assert_array_equal(bundle.frames["left"].image_rgb[0, 0], [10, 20, 30])
        self.assertEqual(bundle.timestamp_ns, 350000000)
        bridge.close()
        self.assertTrue(left.stopped)
        self.assertTrue(right.stopped)

    def test_processor_cannot_return_truth_fields(self):
        frame = frame_from_carla_image(fake_image(1), "episode-1", "front")
        bundle = SensorFrameBundle("episode-1", 1, frame.timestamp_ns, {"front": frame})
        runtime = PerceptionRuntime({
            "bad": lambda current: {"traffic_light_ground_truth": []},
        })
        result = runtime.process(bundle)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["bad"]["status"], "error")
        self.assertEqual(result["source"], "camera_model")

    def test_truth_and_perception_are_separate_files(self):
        frame = frame_from_carla_image(fake_image(2), "episode-1", "front")
        bundle = SensorFrameBundle("episode-1", 2, frame.timestamp_ns, {"front": frame})
        result = PerceptionRuntime().process(bundle)
        with TemporaryDirectory() as directory:
            with EpisodeLogger(directory) as logger:
                logger.log_perception(result)
                logger.log_truth("episode-1", 2, frame.timestamp_ns, {"actors": []})
            perception = Path(directory, "perception.jsonl").read_text().splitlines()
            truth = Path(directory, "truth.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(perception[0])["source"], "sensor_only")
            self.assertEqual(json.loads(truth[0])["source"], "ground_truth")
            self.assertNotIn("truth", json.loads(perception[0]))


if __name__ == "__main__":
    unittest.main()
