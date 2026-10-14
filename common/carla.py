"""CARLA PythonAPI path handling shared by project scripts."""

import os
from pathlib import Path
import sys


def add_carla_pythonapi(carla_root=None):
    """Add a configured CARLA PythonAPI directory before importing ``carla``."""
    configured = carla_root or os.environ.get("CARLA_ROOT")
    if configured is None:
        return None
    root = Path(configured).expanduser().resolve()
    api_root = root / "PythonAPI"
    carla_root = api_root / "carla"
    dist_root = carla_root / "dist"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    compatible_archives = sorted(dist_root.glob(f"*{python_tag}*.whl"))
    compatible_archives += sorted(
        dist_root.glob(f"*py{sys.version_info.major}.{sys.version_info.minor}*.egg")
    )
    # The archive must precede the source directory. Otherwise the CARLA
    # namespace directory shadows the actual Python extension package.
    candidates = tuple(compatible_archives) + (carla_root, api_root)
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root
