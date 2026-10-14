"""Names shared by the sensor, perception, and truth frame contracts."""

FRAME_SCHEMA_VERSION = "perception-frame-0.1"
TRUTH_SCHEMA_VERSION = "scenario-truth-frame-0.1"

FRAME_ID_SOURCE = "carla_sensor"
TRUTH_FRAME_ID_SOURCE = "carla_snapshot"
TIMESTAMP_SOURCE = "carla_simulation"

IMAGE_COLOR_ORDER = "RGB"
IMAGE_DTYPE = "uint8"
IMAGE_COORDINATE_FRAME = "image_pixels_x_right_y_down"
EGO_COORDINATE_FRAME = "ego_vehicle_x_forward_y_right_z_up"

GROUND_TRUTH_SOURCE = "ground_truth"
CAMERA_MODEL_SOURCE = "camera_model"

