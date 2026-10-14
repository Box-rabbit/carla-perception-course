from pathlib import Path

import cv2
import numpy as np

from model_assets import DEFAULT_MODEL_DIR, LABEL_NAME, MODEL_NAME, verify_asset


EXPECTED_LABELS = (
    "green",
    "left,red",
    "left,red,straight",
    "red",
    "red,right",
    "red,straight",
    "unknown",
    "yellow",
    "red,up_left",
    "red,right,straight",
    "red,up_right",
)
SINGLE_COLOR_LABELS = {"red", "yellow", "green"}
MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def preprocess_rgb(rgb_roi):
    if not isinstance(rgb_roi, np.ndarray) or rgb_roi.dtype != np.uint8:
        raise ValueError("Input must be a uint8 RGB NumPy array")
    if rgb_roi.ndim != 3 or rgb_roi.shape[2] != 3 or min(rgb_roi.shape[:2]) <= 0:
        raise ValueError("Input shape must be nonempty H x W x 3")

    height, width = rgb_roi.shape[:2]
    scale = min(224 / width, 224 / height)
    resized_width = max(1, min(224, int(np.rint(width * scale))))
    resized_height = max(1, min(224, int(np.rint(height * scale))))
    resized = cv2.resize(
        rgb_roi,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.zeros((224, 224, 3), dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    normalized = (padded - MEAN_RGB) / STD_RGB
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def decode_probabilities(probabilities, min_confidence=0.7):
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be finite and between 0 and 1")

    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probabilities.size != len(EXPECTED_LABELS):
        raise ValueError("Expected exactly 11 model probabilities")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Model returned invalid probabilities")
    if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-3, rtol=0):
        raise ValueError(
            "Model output must already be softmax probabilities; "
            "do not apply a second softmax"
        )

    label_index = int(np.argmax(probabilities))
    raw_label = EXPECTED_LABELS[label_index]
    confidence = float(probabilities[label_index])

    if confidence < min_confidence:
        state, reason = "unknown", "low_confidence"
    elif raw_label == "unknown":
        state, reason = "unknown", "model_unknown"
    elif raw_label in SINGLE_COLOR_LABELS:
        state, reason = raw_label, "single_color_prediction"
    else:
        state, reason = "unknown", "mixed_lamp_states"

    top_indices = np.argsort(-probabilities, kind="stable")[:3]
    return {
        "state": state,
        "model_confidence": confidence,
        "reason": reason,
        "min_confidence": float(min_confidence),
        "model_debug": {
            "raw_label": raw_label,
            "top3": [
                {
                    "label": EXPECTED_LABELS[int(index)],
                    "probability": float(probabilities[index]),
                }
                for index in top_indices
            ],
        },
    }


class TrafficLightClassifier:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, min_confidence=0.7):
        if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be finite and between 0 and 1")

        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_NAME
        label_path = model_dir / LABEL_NAME
        verify_asset(model_path)
        verify_asset(label_path)
        labels = tuple(label_path.read_text(encoding="utf-8").splitlines())
        if labels != EXPECTED_LABELS:
            raise ValueError("Official labels do not match this decoder")

        self.network = cv2.dnn.readNetFromONNX(str(model_path))
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.min_confidence = float(min_confidence)

    def classify(self, rgb_roi):
        tensor = preprocess_rgb(rgb_roi)
        self.network.setInput(tensor)
        output = self.network.forward()
        return decode_probabilities(output, self.min_confidence)
