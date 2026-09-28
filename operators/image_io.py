"""Unicode-safe OpenCV file IO; failed writes always raise."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if not data.size:
        raise ValueError(f"Empty image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    path.write_bytes(data.tobytes())
    return True
