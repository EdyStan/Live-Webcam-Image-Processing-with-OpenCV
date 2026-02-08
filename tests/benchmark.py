import os
import platform
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import RotatedCropper, rotated_crop


RESOLUTIONS = {
    "SD  640x480":   (480, 640),
    "HD  1280x720":  (720, 1280),
    "FHD 1920x1080": (1080, 1920),
}

TARGETS = {
    "SD  640x480":   60,
    "HD  1280x720":  45,
    "FHD 1920x1080": 30,
}

CROP_CONFIG = {'alpha': 30, 'ox': 0.5, 'oy': 0.5, 'width': 0.4, 'height': 0.4}
WARMUP = 50
ITERATIONS = 1000


def bench(func, warmup=WARMUP, n=ITERATIONS):
    for _ in range(warmup):
        func()
    start = time.perf_counter()
    for _ in range(n):
        func()
    return n / (time.perf_counter() - start)


def main():
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()} | "
          f"{os.cpu_count()} cores | NumPy {np.__version__} | OpenCV {cv2.__version__}\n")

    print("Crop (cached transform)")
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cropper = RotatedCropper(CROP_CONFIG)
        cropper.crop(frame)
        fps = bench(lambda: cropper.crop(frame))
        target = TARGETS[label]
        status = "PASS" if fps >= target else "FAIL"
        print(f"  {label}  {fps:>7.0f} FPS (target {target})  {status}")

    print("\nCrop (no cache)")
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cw, ch = int(w * 0.4), int(h * 0.4)
        fps = bench(lambda: rotated_crop(frame, cx, cy, cw, ch, 30))
        print(f"  {label}  {fps:>7.0f} FPS")

    print("\nFull pipeline (deserialize + crop)")
    header_size = struct.calcsize('diii')
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        data = struct.pack('diii', 0.0, h, w, 3) + frame.tobytes()
        cropper = RotatedCropper(CROP_CONFIG)
        cropper.crop(frame)

        def pipeline():
            _, dh, dw, dc = struct.unpack('diii', data[:header_size])
            f = np.frombuffer(data[header_size:], dtype=np.uint8).reshape((dh, dw, dc))
            return cropper.crop(f)

        fps = bench(pipeline)
        target = TARGETS[label]
        status = "PASS" if fps >= target else "FAIL"
        print(f"  {label}  {fps:>7.0f} FPS (target {target})  {status}")


if __name__ == "__main__":
    main()
