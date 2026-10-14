import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen


MODEL_NAME = "traffic_light_classifier_mobilenetv2_batch_1.onnx"
LABEL_NAME = "lamp_labels.txt"
BASE_URL = "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v2/"
ASSETS = {
    MODEL_NAME: (8926508, "455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703"),
    LABEL_NAME: (118, "1a5a49eeec5593963eab8d70f48b8a01bfb07e753e9688eb1510ad26e803579d"),
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


def verify_asset(path):
    path = Path(path)
    if path.name not in ASSETS:
        raise ValueError(f"Unrecognized official asset name: {path.name}")
    expected_size, expected_hash = ASSETS[path.name]
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"SHA256 mismatch: {path}")
    return digest.hexdigest()


def download_assets(model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        destination = model_dir / name
        if destination.exists():
            verify_asset(destination)
            print(f"Verified existing asset: {destination.resolve()}")
            continue
        temporary = destination.with_name(name + ".part")
        if temporary.exists():
            raise FileExistsError(f"Inspect or remove the incomplete download first: {temporary}")
        expected_size, expected_hash = ASSETS[name]
        digest = hashlib.sha256()
        received = 0
        try:
            with urlopen(BASE_URL + name, timeout=60) as response, temporary.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected_size:
                        raise ValueError(f"Download exceeds expected size: {name}")
                    digest.update(chunk)
                    stream.write(chunk)
            if received != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError(f"Downloaded asset failed verification: {name}")
            if destination.exists():
                raise FileExistsError(f"Destination appeared during download: {destination}")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Downloaded and verified: {destination.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Download pinned official Autoware v2 classifier assets")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    arguments = parser.parse_args()
    download_assets(arguments.model_dir)


if __name__ == "__main__":
    main()
