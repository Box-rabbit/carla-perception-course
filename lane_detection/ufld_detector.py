import hashlib
import importlib
import inspect
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UFLD_DIR = PROJECT_ROOT / "ufld"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "culane_18.pth"
OFFICIAL_COMMIT = "353df107756b8c03c22c27201e33fc63d84ecfe6"
NETWORK_HEIGHT = 288
NETWORK_WIDTH = 800
ROW_ANCHORS = np.array(
    [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
    dtype=np.int32,
)
LANE_COUNT = 4
GRID_SIZE = 200


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if not isinstance(name, str) or not name.isdigit():
        raise ValueError("device must be cpu or a CUDA index such as 0")
    index = int(name)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {name} is unavailable; "
            "use --device cpu or verify CUDA in a normal Linux terminal"
        )
    return torch.device(f"cuda:{index}")


def _ufld_model_class(ufld_dir):
    ufld_dir = Path(ufld_dir).resolve()
    model_file = ufld_dir / "model" / "model.py"
    backbone_file = ufld_dir / "model" / "backbone.py"
    if not model_file.is_file() or not backbone_file.is_file():
        raise FileNotFoundError(f"Official UFLD model code not found: {ufld_dir}")

    existing_model = sys.modules.get("model")
    if existing_model is not None:
        model_paths = [Path(path).resolve() for path in getattr(existing_model, "__path__", [])]
        if model_paths and ufld_dir / "model" not in model_paths:
            raise ImportError(
                "A different top-level Python package named 'model' is already loaded; "
                "refusing ambiguous UFLD imports"
            )
    if str(ufld_dir) not in sys.path:
        sys.path.insert(0, str(ufld_dir))
    module = importlib.import_module("model.model")
    module_file = Path(module.__file__).resolve()
    if module_file != model_file:
        raise ImportError(f"Imported unexpected UFLD model module: {module_file}")
    return module.parsingNet


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(weights, trusted_legacy_checkpoint):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = not trusted_legacy_checkpoint
    checkpoint = torch.load(str(weights), **load_kwargs)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected official UFLD v1 checkpoint with a model state dictionary")
    cleaned = {}
    ignored_aux_keys = []
    for key, value in checkpoint["model"].items():
        if not isinstance(key, str):
            raise ValueError("Invalid checkpoint parameter name")
        name = key[len("module."):] if key.startswith("module.") else key
        if name.startswith(("aux_header2.", "aux_header3.", "aux_header4.", "aux_combine.")):
            ignored_aux_keys.append(name)
            continue
        cleaned[name] = value
    return cleaned, len(ignored_aux_keys)


def _validate_metadata(metadata):
    if metadata is None:
        return {
            "episode_id": None,
            "camera_id": None,
            "frame_id": None,
            "timestamp_ns": None,
            "frame_id_source": None,
            "timestamp_source": None,
        }
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None")
    allowed = {
        "episode_id", "camera_id", "frame_id", "timestamp_ns",
        "frame_id_source", "timestamp_source",
    }
    unknown = set(metadata) - allowed
    if unknown:
        raise ValueError(f"Unsupported metadata fields: {sorted(unknown)}")

    result = {key: metadata.get(key) for key in allowed}
    for key in ("episode_id", "camera_id", "frame_id_source", "timestamp_source"):
        value = result[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string or null")
    for key in ("frame_id", "timestamp_ns"):
        value = result[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{key} must be a nonnegative integer or null")
    return result


class UFLDDetector:
    def __init__(
        self,
        ufld_dir=DEFAULT_UFLD_DIR,
        weights=DEFAULT_WEIGHTS,
        device="0",
        trusted_legacy_checkpoint=False,
        warmup=True,
    ):
        self.ufld_dir = Path(ufld_dir).expanduser().resolve()
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"CULane checkpoint not found: {self.weights}")
        self.device = get_device(device)
        self.device_requested = str(device)
        self.trusted_legacy_checkpoint = bool(trusted_legacy_checkpoint)
        self.weights_sha256 = _sha256(self.weights)
        self.reference_commit = OFFICIAL_COMMIT
        self.ufld_commit_actual = None
        self.ignored_aux_parameter_count = 0

        started = time.perf_counter()
        parsing_net = _ufld_model_class(self.ufld_dir)
        self.model = parsing_net(
            pretrained=False, backbone="18", cls_dim=(GRID_SIZE + 1, len(ROW_ANCHORS), LANE_COUNT),
            use_aux=False,
        )
        state_dict, self.ignored_aux_parameter_count = _load_checkpoint(
            self.weights, self.trusted_legacy_checkpoint
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((NETWORK_HEIGHT, NETWORK_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self._synchronize()
        self.model_load_time_ms = (time.perf_counter() - started) * 1000
        self.warmup_time_ms = 0.0
        if warmup:
            warmup_started = time.perf_counter()
            self._warmup()
            self.warmup_time_ms = (time.perf_counter() - warmup_started) * 1000
        self.initialization_time_ms = self.model_load_time_ms + self.warmup_time_ms

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _warmup(self):
        tensor = torch.zeros((1, 3, NETWORK_HEIGHT, NETWORK_WIDTH), device=self.device)
        with torch.inference_mode():
            self.model(tensor)
        self._synchronize()

    @staticmethod
    def _validate_rgb(image_rgb):
        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
            raise TypeError("image_rgb must be a uint8 NumPy array")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape H x W x 3")
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            raise ValueError("image_rgb must be non-empty")

    def _preprocess(self, image_rgb):
        started = time.perf_counter()
        tensor = self.transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        self._synchronize()
        return tensor, (time.perf_counter() - started) * 1000

    def _decode(self, logits, width, height):
        logits = np.asarray(logits, dtype=np.float64)
        expected_shape = (GRID_SIZE + 1, len(ROW_ANCHORS), LANE_COUNT)
        if logits.shape != expected_shape or not np.isfinite(logits).all():
            raise ValueError(f"Expected finite CULane v1 logits with shape {expected_shape}")
        reversed_logits = logits[:, ::-1, :]
        location_logits = reversed_logits[:-1]
        exponentials = np.exp(location_logits - location_logits.max(axis=0, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=0, keepdims=True)
        grid_indices = np.arange(1, GRID_SIZE + 1, dtype=np.float64).reshape(-1, 1, 1)
        locations = np.sum(probabilities * grid_indices, axis=0)
        present = np.argmax(reversed_logits, axis=0) != GRID_SIZE
        full_exponentials = np.exp(reversed_logits - reversed_logits.max(axis=0, keepdims=True))
        full_probabilities = full_exponentials / full_exponentials.sum(axis=0, keepdims=True)

        lanes = []
        column_spacing = (NETWORK_WIDTH - 1) / (GRID_SIZE - 1)
        for slot_index in range(LANE_COUNT):
            points = []
            row_presence = []
            for anchor_index, anchor_y in enumerate(ROW_ANCHORS[::-1]):
                row_presence.append(float(1 - full_probabilities[-1, anchor_index, slot_index]))
                if not present[anchor_index, slot_index]:
                    continue
                pixel_x = int(locations[anchor_index, slot_index] * column_spacing * width / NETWORK_WIDTH) - 1
                pixel_y = int(height * anchor_y / NETWORK_HEIGHT) - 1
                if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                    continue
                points.append({
                    "xy_px": [pixel_x, pixel_y],
                    "sample_row_index_top_to_bottom": len(ROW_ANCHORS) - 1 - anchor_index,
                    "anchor_index_bottom_to_top": anchor_index,
                    "sample_row_y_network_px": int(anchor_y),
                })
            lanes.append({
                "lane_id": f"lane_{slot_index + 1}",
                "model_slot": slot_index,
                "valid": len(points) >= 3,
                "point_count": len(points),
                "points": points,
                "row_presence_probabilities_bottom_to_top": row_presence,
                "ego_boundary_role": "unknown",
            })
        return lanes

    def detect(self, image_rgb, metadata=None):
        self._validate_rgb(image_rgb)
        frame_metadata = _validate_metadata(metadata)
        height, width = image_rgb.shape[:2]
        tensor, preprocess_ms = self._preprocess(image_rgb)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        self._synchronize()
        forward_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        logits = output[0].detach().cpu().numpy()
        lanes = self._decode(logits, width, height)
        transfer_decode_ms = (time.perf_counter() - started) * 1000
        return {
            "schema_version": "lane2d-0.2",
            "status": "ok",
            "lanes_valid": True,
            "metadata": frame_metadata,
            "image_width_px": int(width),
            "image_height_px": int(height),
            "coordinate_frame": "image_pixels_x_right_y_down",
            "model": "UFLD_v1_CULane_ResNet18",
            "device_actual": str(self.device),
            "valid_lane_count": sum(lane["valid"] for lane in lanes),
            "lanes": lanes,
            "errors": [],
            "ego_boundaries_valid": False,
            "left_boundary_lane_id": None,
            "right_boundary_lane_id": None,
            "ego_boundary_role": "unknown",
            "ego_boundary_method": "unavailable_missing_calibration",
            "metric_distances_valid": False,
            "left_distance_m": None,
            "right_distance_m": None,
            "metric_distance_reason": "missing_camera_calibration_vehicle_reference_and_road_geometry",
            "timings_ms": {
                "preprocess_ms": preprocess_ms,
                "forward_ms": forward_ms,
                "transfer_and_decode_ms": transfer_decode_ms,
            },
        }
