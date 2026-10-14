from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from demo_classifier import classify_image, source_images
from model_assets import ASSETS, verify_asset
from traffic_light_classifier import EXPECTED_LABELS, MEAN_RGB, STD_RGB, decode_probabilities, preprocess_rgb


def probabilities_for(label, confidence=1.0):
    probabilities = np.full(11, (1.0 - confidence) / 10, dtype=np.float32)
    probabilities[EXPECTED_LABELS.index(label)] = confidence
    return probabilities


class ClassifierTests(unittest.TestCase):
    def test_circle_colors(self):
        for color in ("red", "yellow", "green"):
            with self.subTest(color=color):
                result = decode_probabilities(probabilities_for(color))
                self.assertEqual(result["state"], color)
                self.assertEqual(result["reason"], "single_color_prediction")
                self.assertEqual(result["model_debug"]["raw_label"], color)

    def test_mixed_labels_are_not_red_arrows(self):
        for label in EXPECTED_LABELS:
            if "," not in label:
                continue
            with self.subTest(label=label):
                result = decode_probabilities(probabilities_for(label))
                self.assertEqual(result["state"], "unknown")
                self.assertEqual(result["reason"], "mixed_lamp_states")
                self.assertEqual(result["model_debug"]["raw_label"], label)

    def test_low_confidence(self):
        result = decode_probabilities(probabilities_for("red", 0.5))
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["reason"], "low_confidence")
        self.assertAlmostEqual(result["model_confidence"], 0.5)

    def test_unknown(self):
        result = decode_probabilities(probabilities_for("unknown"))
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["reason"], "model_unknown")

    def test_invalid_model_outputs(self):
        for probabilities in (np.zeros(11), np.ones(10), np.full(11, np.nan), np.full(11, -0.1)):
            with self.subTest(probabilities=str(probabilities)):
                with self.assertRaises(ValueError):
                    decode_probabilities(probabilities)

    def test_invalid_threshold(self):
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                decode_probabilities(probabilities_for("red"), threshold)

    def test_no_second_softmax(self):
        result = decode_probabilities(probabilities_for("green", 0.9))
        self.assertAlmostEqual(result["model_confidence"], 0.9, places=6)

    def test_top_left_padding_and_rgb_normalization(self):
        rgb = np.full((4, 2, 3), (255, 100, 10), dtype=np.uint8)
        tensor = preprocess_rgb(rgb)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], (np.array([255, 100, 10]) - MEAN_RGB) / STD_RGB)
        np.testing.assert_allclose(tensor[0, :, 0, 111], tensor[0, :, 0, 0])
        np.testing.assert_allclose(tensor[0, :, 0, 112], -MEAN_RGB / STD_RGB)

    def test_wide_image_pads_bottom(self):
        tensor = preprocess_rgb(np.full((2, 4, 3), 255, dtype=np.uint8))
        np.testing.assert_allclose(tensor[0, :, 112, 0], -MEAN_RGB / STD_RGB)
        self.assertGreater(float(tensor[0, 0, 111, 0]), 0)

    def test_invalid_rgb(self):
        for rgb in (None, np.zeros((0, 3, 3), np.uint8), np.zeros((2, 2), np.uint8),
                    np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32)):
            with self.assertRaises(ValueError):
                preprocess_rgb(rgb)

    def test_roi_and_bgr_to_rgb(self):
        class RecordingClassifier:
            def classify(self, rgb):
                np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
                self_shape = rgb.shape
                if self_shape != (2, 2, 3):
                    raise AssertionError(str(self_shape))
                return {"state": "red"}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.png"
            cv2.imwrite(str(path), np.full((4, 4, 3), [10, 20, 30], np.uint8))
            record = classify_image(RecordingClassifier(), path, (1, 1, 3, 3))
            self.assertEqual(record["roi_xyxy_px"], [1, 1, 3, 3])
            self.assertEqual(record["state"], "red")
            with self.assertRaises(ValueError):
                classify_image(RecordingClassifier(), path, (-1, 0, 2, 2))

    def test_empty_source_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                source_images(folder)

    def test_asset_hash_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lamp_labels.txt"
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                verify_asset(path)
            with patch.dict(ASSETS, {path.name: (5, "0" * 64)}):
                with self.assertRaises(ValueError):
                    verify_asset(path)


if __name__ == "__main__":
    unittest.main()
