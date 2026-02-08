"""
Full end-to-end pipeline benchmark.

Measures: frame generation -> serialization -> ZeroMQ PUB/SUB -> deserialization -> clamp + crop.
Uses synthetic frames (no webcam needed), no display.
"""

import os
import platform
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import RotatedCropper

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

CLAMP_CONFIGS = {
    "In bounds":     {"alpha": 353.34, "ox": 0.434, "oy": 0.493,
                      "width": 0.241, "height": 0.627},
    "1 corner out":  {"alpha": 20, "ox": 0.10, "oy": 0.25,
                      "width": 0.22, "height": 0.18},
    "2 corners out": {"alpha": 30, "ox": 0.85, "oy": 0.15,
                      "width": 0.3, "height": 0.2},
}

HEADER_FMT = 'diii'
HEADER_SIZE = struct.calcsize(HEADER_FMT)
WARMUP_SEC = 1.0
BENCH_SEC = 5.0
BASE_PORT = 15555


def publisher_thread(port, frame, duration):
    """Publish a pre-serialized synthetic frame in a tight loop."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{port}")

    h, w = frame.shape[:2]
    c = frame.shape[2] if len(frame.shape) == 3 else 1
    payload = struct.pack(HEADER_FMT, 0.0, h, w, c) + frame.tobytes()

    # Give subscriber time to connect
    time.sleep(0.3)

    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        sock.send(payload, zmq.NOBLOCK)

    sock.close()
    ctx.term()


def run_benchmark(port, frame, cropper, warmup_sec, bench_sec):
    """Subscribe, deserialize, crop - return FPS over the bench window."""
    total_duration = warmup_sec + bench_sec

    pub = threading.Thread(
        target=publisher_thread,
        args=(port, frame, total_duration + 1.0),
        daemon=True,
    )
    pub.start()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://localhost:{port}")

    # Warmup
    warmup_deadline = time.perf_counter() + warmup_sec
    while time.perf_counter() < warmup_deadline:
        if sock.poll(timeout=100):
            data = sock.recv()
            _, h, w, c = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
            f = np.frombuffer(data[HEADER_SIZE:], dtype=np.uint8).reshape((h, w, c))
            cropper.crop(f)

    # Benchmark
    count = 0
    bench_deadline = time.perf_counter() + bench_sec
    while time.perf_counter() < bench_deadline:
        if sock.poll(timeout=100):
            data = sock.recv()
            _, h, w, c = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
            f = np.frombuffer(data[HEADER_SIZE:], dtype=np.uint8).reshape((h, w, c))
            cropper.crop(f)
            count += 1

    sock.close()
    ctx.term()
    pub.join(timeout=3)

    return count / bench_sec if bench_sec > 0 else 0


def main():
    print(f"Python {platform.python_version()} | "
          f"{platform.system()} {platform.machine()} | "
          f"{os.cpu_count()} cores | "
          f"NumPy {np.__version__} | OpenCV {cv2.__version__}\n")

    print("Full pipeline (ZMQ PUB/SUB -> deserialize -> clamp + crop)")
    print(f"  warmup {WARMUP_SEC:.0f}s, measure {BENCH_SEC:.0f}s per run\n")

    port = BASE_PORT
    for clamp_label, cfg in CLAMP_CONFIGS.items():
        for res_label, (h, w) in RESOLUTIONS.items():
            frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            cropper = RotatedCropper(cfg)
            # Prime the cropper once so first-call setup cost is excluded
            cropper.crop(frame)

            fps = run_benchmark(port, frame, cropper, WARMUP_SEC, BENCH_SEC)
            target = TARGETS[res_label]
            status = "PASS" if fps >= target else "FAIL"
            print(f"  {clamp_label:<16s} {res_label}  {fps:>7.0f} FPS "
                  f"(target {target})  {status}")
            port += 1
        print()


if __name__ == "__main__":
    main()
