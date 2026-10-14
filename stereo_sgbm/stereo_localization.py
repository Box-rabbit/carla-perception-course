import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def finite_array(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def load_calibration(path):
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != "stereo-pinhole-calibration-0.1":
        raise ValueError("Unsupported calibration schema")
    if payload.get("camera_model") != "opencv_pinhole":
        raise ValueError("This entry supports OpenCV pinhole calibration, not fisheye calibration")
    if payload.get("translation_unit") != "m":
        raise ValueError("Translation must be supplied in metres, with translation_unit=m")
    if payload.get("extrinsic_convention") != "X_right=R_left_to_right@X_left+T_left_to_right_m":
        raise ValueError("Extrinsic convention must explicitly map left-camera points to right-camera points")
    size = payload.get("image_size_px")
    if (
        not isinstance(size, list) or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError("image_size_px must be [width, height] with positive integers")
    matrices = {}
    for name in ("K_left", "K_right"):
        matrix = finite_array(payload.get(name), (3, 3), name)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"{name} focal lengths must be positive")
        if not np.allclose(matrix[2], [0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError(f"{name} must have last row [0, 0, 1]")
        if abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8:
            raise ValueError(f"{name} must use the zero-skew OpenCV pinhole convention")
        matrices[name] = matrix
    distortions = {}
    for name in ("D_left", "D_right"):
        distortion = np.asarray(payload.get(name), dtype=np.float64)
        if distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"{name} must be a 4/5/8/12/14-element OpenCV distortion vector")
        if not np.isfinite(distortion).all():
            raise ValueError(f"{name} contains nonfinite values")
        distortions[name] = distortion
    rotation = finite_array(payload.get("R_left_to_right"), (3, 3), "R_left_to_right")
    if (
        not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("R_left_to_right must be a proper rotation matrix")
    translation = finite_array(payload.get("T_left_to_right_m"), (3,), "T_left_to_right_m")
    baseline_m = float(np.linalg.norm(translation))
    if baseline_m <= 1e-6:
        raise ValueError("A nonzero measured or simulator-derived stereo baseline is required")
    return {
        **matrices,
        **distortions,
        "rotation": rotation,
        "translation": translation,
        "image_size": tuple(size),
        "baseline_m": baseline_m,
        "source_description": payload.get("source_description"),
    }


def roi_mask(shape, roi):
    mask = np.zeros(shape, dtype=bool)
    left, top, width, height = roi
    mask[top:top + height, left:left + width] = True
    return mask


class StereoLocalizer:
    def __init__(self, calibration, num_disparities=128, block_size=5,
                 min_disparity=0, min_depth_m=0.5, max_depth_m=80.0, lr_threshold_px=1.0):
        if type(num_disparities) is not int or num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if type(block_size) is not int or block_size < 3 or block_size > 21 or block_size % 2 == 0:
            raise ValueError("block_size must be odd, between 3 and 21")
        if type(min_disparity) is not int or min_disparity < 0:
            raise ValueError("This positive-disparity entry requires min_disparity >= 0")
        if (
            not np.isfinite([min_depth_m, max_depth_m, lr_threshold_px]).all()
            or not 0 < min_depth_m < max_depth_m or lr_threshold_px <= 0
        ):
            raise ValueError("Depth limits and left-right consistency threshold are invalid")
        width, height = calibration["image_size"]
        if width <= min_disparity + num_disparities + block_size:
            raise ValueError("Image width is too small for the configured disparity range")
        self.calibration = calibration
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.lr_threshold_px = float(lr_threshold_px)
        rectification = cv2.stereoRectify(
            calibration["K_left"], calibration["D_left"],
            calibration["K_right"], calibration["D_right"],
            calibration["image_size"], calibration["rotation"], calibration["translation"],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.rotation_left, self.rotation_right, self.projection_left, self.projection_right = rectification[:4]
        self.q_matrix = rectification[4]
        self.roi_left, self.roi_right = rectification[5:7]
        if (
            abs(self.projection_right[1, 3]) > 1e-6
            or self.projection_right[0, 3] >= -1e-6
        ):
            raise ValueError("Requires horizontal stereo with the right camera on the right and positive x_left-x_right disparity")
        self.map_left = cv2.initUndistortRectifyMap(
            calibration["K_left"], calibration["D_left"], self.rotation_left,
            self.projection_left, calibration["image_size"], cv2.CV_32FC1,
        )
        self.map_right = cv2.initUndistortRectifyMap(
            calibration["K_right"], calibration["D_right"], self.rotation_right,
            self.projection_right, calibration["image_size"], cv2.CV_32FC1,
        )
        original_mask = np.ones((height, width), dtype=np.float32)
        left_coverage = cv2.remap(original_mask, *self.map_left, cv2.INTER_LINEAR)
        right_coverage = cv2.remap(original_mask, *self.map_right, cv2.INTER_LINEAR)
        self.left_coverage = (left_coverage >= 0.999) & roi_mask((height, width), self.roi_left)
        self.right_coverage = (right_coverage >= 0.999) & roi_mask((height, width), self.roi_right)
        valid_roi = cv2.getValidDisparityROI(
            self.roi_left, self.roi_right, min_disparity, num_disparities, block_size,
        )
        self.matching_coverage = roi_mask((height, width), valid_roi)
        common = {
            "numDisparities": num_disparities,
            "blockSize": block_size,
            "P1": 8 * block_size * block_size,
            "P2": 32 * block_size * block_size,
            "disp12MaxDiff": 1,
            "preFilterCap": 31,
            "uniquenessRatio": 10,
            "speckleWindowSize": 100,
            "speckleRange": 2,
            "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
        self.matcher_left = cv2.StereoSGBM_create(minDisparity=min_disparity, **common)
        self.min_disparity_right = -min_disparity - num_disparities + 1
        self.matcher_right = cv2.StereoSGBM_create(minDisparity=self.min_disparity_right, **common)

    def process(self, left_bgr, right_bgr):
        width, height = self.calibration["image_size"]
        for name, image in (("left", left_bgr), ("right", right_bgr)):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError(f"{name} input must be a uint8 NumPy image")
            if image.shape != (height, width, 3):
                raise ValueError(f"{name} image must match calibrated H x W x 3: {(height, width, 3)}")
        start = time.perf_counter()
        left_rectified = cv2.remap(left_bgr, *self.map_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_bgr, *self.map_right, cv2.INTER_LINEAR)
        rectification_ms = (time.perf_counter() - start) * 1000
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        start = time.perf_counter()
        fixed_left = self.matcher_left.compute(left_gray, right_gray)
        fixed_right = self.matcher_right.compute(right_gray, left_gray)
        matching_ms = (time.perf_counter() - start) * 1000
        disparity = fixed_left.astype(np.float32) / 16.0
        disparity_right = fixed_right.astype(np.float32) / 16.0
        start = time.perf_counter()
        pixel_rows, pixel_columns = np.indices((height, width))
        right_columns = np.rint(pixel_columns - disparity).astype(np.int32)
        in_bounds = (right_columns >= 0) & (right_columns < width)
        safe_columns = np.clip(right_columns, 0, width - 1)
        sampled_right = disparity_right[pixel_rows, safe_columns]
        sampled_right_fixed = fixed_right[pixel_rows, safe_columns]
        right_valid = sampled_right_fixed > (self.min_disparity_right - 1) * 16
        consistent = np.abs(disparity + sampled_right) <= self.lr_threshold_px
        valid_disparity = (
            (fixed_left > (self.min_disparity - 1) * 16)
            & (disparity > 0)
            & self.left_coverage & self.matching_coverage
            & in_bounds & right_valid
            & self.right_coverage[pixel_rows, safe_columns] & consistent
        )
        xyz = cv2.reprojectImageTo3D(disparity, self.q_matrix, handleMissingValues=False)
        valid_depth = (
            valid_disparity & np.isfinite(xyz).all(axis=2)
            & (xyz[:, :, 2] >= self.min_depth_m) & (xyz[:, :, 2] <= self.max_depth_m)
        )
        filtered_disparity = disparity.copy()
        filtered_disparity[~valid_disparity] = np.nan
        xyz[~valid_depth] = np.nan
        depth = xyz[:, :, 2].copy()
        postprocess_ms = (time.perf_counter() - start) * 1000
        return {
            "left_rectified": left_rectified,
            "right_rectified": right_rectified,
            "disparity_px": filtered_disparity,
            "depth_m": depth,
            "xyz_m": xyz,
            "valid_disparity_mask": valid_disparity,
            "valid_depth_mask": valid_depth,
            "stage_times_ms": {
                "rectification_ms": rectification_ms,
                "sgbm_left_and_right_ms": matching_ms,
                "consistency_and_reprojection_ms": postprocess_ms,
            },
        }


def colorize(values, valid, minimum, maximum):
    display = np.zeros(values.shape, dtype=np.uint8)
    display[valid] = np.clip((values[valid] - minimum) / (maximum - minimum) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Image save failed: {path}")


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def make_run_dir(root, sample_id):
    module_dir = root / "stereo_3d"
    module_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_" + sample_id
    suffix = 0
    while True:
        path = module_dir / (name if suffix == 0 else f"{name}_{suffix:02d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            suffix += 1


def main():
    parser = argparse.ArgumentParser(description="Calibrated horizontal stereo SGBM localization, CPU")
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--check-calibration", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--min-depth-m", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--lr-threshold-px", type=float, default=1.0)
    parser.add_argument("--probe", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    arguments = parser.parse_args()
    if arguments.check_calibration:
        if arguments.left is not None or arguments.right is not None:
            parser.error("Use --check-calibration separately from image inputs")
    elif arguments.left is None or arguments.right is None:
        parser.error("Provide both --left and --right")
    sample_id = arguments.sample_id or (arguments.left.stem if arguments.left else "calibration")
    if sample_id in (".", "..") or any(character in sample_id for character in "/\\\x00"):
        parser.error("--sample-id must be a plain filename-like identifier")
    run_dir = None
    start = time.perf_counter()
    run = {
        "schema_version": "sgbm-stereo-run-0.1",
        "status": "running",
        "started_local": datetime.now().astimezone().isoformat(),
        "device_actual": "cpu",
        "left_source": str(arguments.left.resolve()) if arguments.left else None,
        "right_source": str(arguments.right.resolve()) if arguments.right else None,
        "calibration_source": str(arguments.calibration.resolve()),
        "synchronization_verified_by_program": False,
        "parameters": {
            "num_disparities": arguments.num_disparities,
            "block_size": arguments.block_size,
            "min_disparity": arguments.min_disparity,
            "min_depth_m": arguments.min_depth_m,
            "max_depth_m": arguments.max_depth_m,
            "lr_threshold_px": arguments.lr_threshold_px,
        },
        "error_summary": [],
    }
    try:
        if not arguments.check_calibration:
            run_dir = make_run_dir(arguments.output_root.resolve(), sample_id)
            save_json(run_dir / "run.json", run)
        calibration = load_calibration(arguments.calibration)
        localizer = StereoLocalizer(
            calibration,
            arguments.num_disparities, arguments.block_size, arguments.min_disparity,
            arguments.min_depth_m, arguments.max_depth_m, arguments.lr_threshold_px,
        )
        if arguments.check_calibration:
            print(f"Calibration structure check passed: size={calibration['image_size']}, baseline={calibration['baseline_m']:.6f} m")
            print("This does not verify calibration accuracy or image synchronization.")
            return
        sample = run_dir / "samples" / sample_id
        for name in ("00_input", "01_rectification", "02_disparity", "03_result"):
            (sample / name).mkdir(parents=True)
        left = cv2.imread(str(arguments.left.resolve()), cv2.IMREAD_COLOR)
        right = cv2.imread(str(arguments.right.resolve()), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise ValueError("Left or right image cannot be decoded")
        result = localizer.process(left, right)
        save_image(sample / "00_input" / "left.png", left)
        save_image(sample / "00_input" / "right.png", right)
        save_json(sample / "00_input" / "calibration.json", json.loads(arguments.calibration.read_text(encoding="utf-8")))
        save_image(sample / "01_rectification" / "left_rectified.png", result["left_rectified"])
        save_image(sample / "01_rectification" / "right_rectified.png", result["right_rectified"])
        epipolar = np.hstack([result["left_rectified"], result["right_rectified"]])
        for pixel_y in range(20, left.shape[0], 40):
            cv2.line(epipolar, (0, pixel_y), (epipolar.shape[1] - 1, pixel_y), (0, 255, 0), 1)
        save_image(sample / "01_rectification" / "epipolar_check.jpg", epipolar)
        save_json(sample / "01_rectification" / "geometry.json", {
            "coordinate_frame": "rectified_left_camera_x_right_y_down_z_forward",
            "R_left_rectification": localizer.rotation_left.tolist(),
            "R_right_rectification": localizer.rotation_right.tolist(),
            "P_left": localizer.projection_left.tolist(),
            "P_right": localizer.projection_right.tolist(),
            "Q": localizer.q_matrix.tolist(),
            "baseline_m": calibration["baseline_m"],
            "xyz_original_left_relation": "X_rectified=R_left_rectification@X_original_left",
        })
        save_image(sample / "02_disparity" / "disparity_color.png", colorize(
            result["disparity_px"], result["valid_disparity_mask"],
            arguments.min_disparity, arguments.min_disparity + arguments.num_disparities,
        ))
        save_image(sample / "02_disparity" / "valid_disparity_mask.png", result["valid_disparity_mask"].astype(np.uint8) * 255)
        np.save(sample / "02_disparity" / "disparity_px.npy", result["disparity_px"], allow_pickle=False)
        save_image(sample / "03_result" / "depth_color.png", colorize(
            result["depth_m"], result["valid_depth_mask"], arguments.min_depth_m, arguments.max_depth_m,
        ))
        save_image(sample / "03_result" / "valid_depth_mask.png", result["valid_depth_mask"].astype(np.uint8) * 255)
        np.save(sample / "03_result" / "depth_m.npy", result["depth_m"], allow_pickle=False)
        np.save(sample / "03_result" / "xyz_m.npy", result["xyz_m"], allow_pickle=False)
        valid_count = int(result["valid_depth_mask"].sum())
        summary = {
            "schema_version": "stereo-localization-0.1",
            "status": "ok",
            "coordinate_frame": "rectified_left_camera_x_right_y_down_z_forward",
            "image_width_px": left.shape[1],
            "image_height_px": left.shape[0],
            "baseline_m": calibration["baseline_m"],
            "valid_disparity_pixel_count": int(result["valid_disparity_mask"].sum()),
            "valid_depth_pixel_count": valid_count,
            "valid_depth_fraction": valid_count / (left.shape[0] * left.shape[1]),
            "has_valid_depth": valid_count > 0,
            "depth_definition": "Z forward along rectified left camera optical axis, metres; not Euclidean range",
            "disparity_definition": "x_left_rectified-x_right_rectified, pixels",
            "invalid_numeric_array_value": "NaN",
            "synchronization_verified_by_program": False,
            "files": {
                "disparity_px": (sample / "02_disparity" / "disparity_px.npy").relative_to(run_dir).as_posix(),
                "depth_m": (sample / "03_result" / "depth_m.npy").relative_to(run_dir).as_posix(),
                "xyz_m": (sample / "03_result" / "xyz_m.npy").relative_to(run_dir).as_posix(),
            },
            "probe": None,
        }
        if arguments.probe:
            pixel_x, pixel_y = arguments.probe
            if not (0 <= pixel_x < left.shape[1] and 0 <= pixel_y < left.shape[0]):
                raise ValueError("--probe must be within the rectified image")
            valid = bool(result["valid_depth_mask"][pixel_y, pixel_x])
            summary["probe"] = {
                "rectified_xy_px": [pixel_x, pixel_y],
                "valid": valid,
                "xyz_m": result["xyz_m"][pixel_y, pixel_x].astype(float).tolist() if valid else None,
            }
        save_json(sample / "03_result" / "result.json", summary)
        run["status"] = "ok"
        run["baseline_m"] = calibration["baseline_m"]
        run["stage_times_ms"] = result["stage_times_ms"]
        run["valid_depth_pixel_count"] = valid_count
        run["final_result"] = (sample / "03_result" / "result.json").relative_to(run_dir).as_posix()
        print(f"Valid depth pixels: {valid_count}; fraction={summary['valid_depth_fraction']:.3f}")
    except Exception as error:
        run["status"] = "error"
        run["error_summary"] = [{"type": type(error).__name__, "message": str(error)}]
        raise
    finally:
        if run_dir is not None:
            run["finished_local"] = datetime.now().astimezone().isoformat()
            run["elapsed_ms"] = (time.perf_counter() - start) * 1000
            save_json(run_dir / "run.json", run)
            print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
