#!/usr/bin/env python3
"""
Performance benchmark for the robotics vision pipeline.

Methodology:
  - Each benchmark uses timeit.repeat (wall-clock via perf_counter)
  - Warm-up phase discarded before measurement
  - Reports: median, mean, std, P5/P95 over multiple runs
  - GC disabled during measurement to reduce noise

Usage:
    python tests/benchmark.py
    python tests/benchmark.py --save results.txt
"""

import gc
import os
import platform
import struct
import sys
import timeit
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import RotatedCropper, rotated_crop


# -- Configuration ------------------------------------------------------------

WARMUP = 20           # warm-up iterations (discarded)
REPEATS = 7           # number of timed rounds
MIN_DURATION = 0.5    # target seconds per round (auto-scales iterations)

RESOLUTIONS = {
    "SD  640x480":   (480, 640),
    "HD  1280x720":  (720, 1280),
    "FHD 1920x1080": (1080, 1920),
}

TARGETS = {
    "SD  640x480":   60,    # FPS
    "HD  1280x720":  45,
    "FHD 1920x1080": 30,
}

CROP_CONFIG = {
    'alpha': 30, 'ox': 0.5, 'oy': 0.5, 'width': 0.4, 'height': 0.4
}


# -- Helpers ------------------------------------------------------------------

def auto_n(func, target_secs=MIN_DURATION):
    """Find iteration count so one round takes ~target_secs."""
    n = 1
    while True:
        start = timeit.default_timer()
        for _ in range(n):
            func()
        elapsed = timeit.default_timer() - start
        if elapsed >= target_secs * 0.5:
            return max(1, int(n * target_secs / elapsed))
        n *= 4


def bench(func, label="", warmup=WARMUP, repeats=REPEATS):
    """Run a benchmark and return per-call timings in seconds."""
    # Warm-up
    for _ in range(warmup):
        func()

    n = auto_n(func)

    # Timed rounds
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        round_times = []
        for _ in range(repeats):
            start = timeit.default_timer()
            for _ in range(n):
                func()
            elapsed = timeit.default_timer() - start
            round_times.append(elapsed / n)
    finally:
        if gc_was_enabled:
            gc.enable()

    return np.array(round_times)


def fmt_time(secs):
    """Format seconds as a human-readable string."""
    if secs < 1e-6:
        return f"{secs * 1e9:.0f} ns"
    elif secs < 1e-3:
        return f"{secs * 1e6:.1f} us"
    elif secs < 1:
        return f"{secs * 1e3:.2f} ms"
    else:
        return f"{secs:.3f} s"


def fmt_fps(secs):
    """Convert per-frame seconds to FPS string."""
    return f"{1.0 / secs:.0f}" if secs > 0 else "inf"


# -- System info --------------------------------------------------------------

def collect_system_info():
    lines = []
    lines.append(f"  Python:       {platform.python_version()} ({platform.python_implementation()})")
    lines.append(f"  Platform:     {platform.system()} {platform.version()}")
    lines.append(f"  Architecture: {platform.machine()}")
    lines.append(f"  CPU cores:    {os.cpu_count()}")
    lines.append(f"  NumPy:        {np.__version__}")
    lines.append(f"  OpenCV:       {cv2.__version__}")

    # Check SIMD from OpenCV build info
    build_info = cv2.getBuildInformation()
    for flag in ("AVX512", "AVX2", "AVX", "SSE4", "NEON"):
        if flag in build_info:
            dispatched = "yes" if f"{flag} (1" in build_info or f"requested: {flag}" in build_info.upper() else "available"
            lines.append(f"  SIMD:         {flag} {dispatched}")
            break

    return lines


# -- Benchmark suites ---------------------------------------------------------

def bench_crop_function():
    """Benchmark the low-level rotated_crop function."""
    results = {}
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cw, ch = int(w * 0.4), int(h * 0.4)

        timings = bench(lambda: rotated_crop(frame, cx, cy, cw, ch, 30))
        results[label] = timings
    return results


def bench_cropper_cached():
    """Benchmark RotatedCropper with cached transform matrix."""
    results = {}
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cropper = RotatedCropper(CROP_CONFIG)
        # Prime the cache
        cropper.crop(frame)

        timings = bench(lambda: cropper.crop(frame))
        results[label] = timings
    return results


def bench_serialization():
    """Benchmark frame serialization (struct.pack + tobytes)."""
    results = {}
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ts = 1234567890.123

        def serialize():
            fh, fw = frame.shape[:2]
            c = frame.shape[2]
            return struct.pack('diii', ts, fh, fw, c) + frame.tobytes()

        timings = bench(serialize)
        results[label] = timings
    return results


def bench_deserialization():
    """Benchmark frame deserialization (struct.unpack + frombuffer)."""
    results = {}
    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        c = frame.shape[2]
        header = struct.pack('diii', 0.0, h, w, c)
        data = header + frame.tobytes()
        header_size = struct.calcsize('diii')

        def deserialize():
            _, dh, dw, dc = struct.unpack('diii', data[:header_size])
            return np.frombuffer(data[header_size:], dtype=np.uint8).reshape((dh, dw, dc))

        timings = bench(deserialize)
        results[label] = timings
    return results


def bench_full_pipeline():
    """Benchmark the full pipeline: serialize -> deserialize -> crop."""
    results = {}
    header_size = struct.calcsize('diii')

    for label, (h, w) in RESOLUTIONS.items():
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        c = frame.shape[2]
        header = struct.pack('diii', 0.0, h, w, c)
        data = header + frame.tobytes()
        cropper = RotatedCropper(CROP_CONFIG)
        cropper.crop(frame)  # prime cache

        def pipeline():
            _, dh, dw, dc = struct.unpack('diii', data[:header_size])
            f = np.frombuffer(data[header_size:], dtype=np.uint8).reshape((dh, dw, dc))
            return cropper.crop(f)

        timings = bench(pipeline)
        results[label] = timings
    return results


# -- Reporting ----------------------------------------------------------------

def print_section(title, results, show_target=False):
    """Print a formatted results table."""
    # Header
    print(f"\n  {'Resolution':<20} {'Median':>10} {'Mean':>10} {'Std':>9} "
          f"{'P5':>10} {'P95':>10} {'FPS':>8}", end="")
    if show_target:
        print(f" {'Target':>8} {'Status':>8}", end="")
    print()
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 9} "
          f"{'-' * 10} {'-' * 10} {'-' * 8}", end="")
    if show_target:
        print(f" {'-' * 8} {'-' * 8}", end="")
    print()

    for label, timings in results.items():
        median = np.median(timings)
        mean = np.mean(timings)
        std = np.std(timings)
        p5 = np.percentile(timings, 5)
        p95 = np.percentile(timings, 95)
        fps = 1.0 / median if median > 0 else float('inf')

        line = (f"  {label:<20} {fmt_time(median):>10} {fmt_time(mean):>10} "
                f"{fmt_time(std):>9} {fmt_time(p5):>10} {fmt_time(p95):>10} "
                f"{fps:>7.0f}")

        if show_target:
            target = TARGETS.get(label, 0)
            passed = fps >= target
            status = "PASS" if passed else "FAIL"
            marker = "[+]" if passed else "[-]"
            line += f" {target:>7} {marker:>5} {status}"

        print(line)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline performance benchmarks")
    parser.add_argument('--save', type=str, default=None,
                        help='Save output to a text file')
    args = parser.parse_args()

    # Capture output to both console and optionally a file
    output_lines = []

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            output_lines.append(text)
        def flush(self):
            sys.__stdout__.flush()

    if args.save:
        sys.stdout = Tee()

    print("=" * 88)
    print("  ROBOTICS VISION PIPELINE - PERFORMANCE BENCHMARK")
    print("=" * 88)

    print("\n  System Information")
    print("  " + "-" * 40)
    for line in collect_system_info():
        print(line)

    print(f"\n  Benchmark settings: {WARMUP} warm-up, {REPEATS} rounds, "
          f"~{MIN_DURATION}s per round")
    print(f"  Crop config: alpha={CROP_CONFIG['alpha']}, "
          f"size={CROP_CONFIG['width']}x{CROP_CONFIG['height']}")

    # -- Run benchmarks ---------------------------------------------------
    print("\n" + "-" * 88)
    print("  1. ROTATED CROP (cached transform)")
    print("-" * 88)
    crop_results = bench_cropper_cached()
    print_section("Rotated Crop (cached)", crop_results, show_target=True)

    print("\n" + "-" * 88)
    print("  2. ROTATED CROP (recompute transform each call)")
    print("-" * 88)
    raw_results = bench_crop_function()
    print_section("Rotated Crop (raw)", raw_results)

    print("\n" + "-" * 88)
    print("  3. FRAME SERIALIZATION (struct.pack + tobytes)")
    print("-" * 88)
    ser_results = bench_serialization()
    print_section("Serialization", ser_results)

    print("\n" + "-" * 88)
    print("  4. FRAME DESERIALIZATION (struct.unpack + frombuffer)")
    print("-" * 88)
    deser_results = bench_deserialization()
    print_section("Deserialization", deser_results)

    print("\n" + "-" * 88)
    print("  5. FULL PIPELINE (deserialize + crop)")
    print("-" * 88)
    pipe_results = bench_full_pipeline()
    print_section("Full Pipeline", pipe_results, show_target=True)

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 88)
    print("  SUMMARY")
    print("=" * 88)

    all_pass = True
    for label in RESOLUTIONS:
        target = TARGETS.get(label, 0)
        crop_fps = 1.0 / np.median(crop_results[label])
        pipe_fps = 1.0 / np.median(pipe_results[label])
        crop_ok = crop_fps >= target
        pipe_ok = pipe_fps >= target
        all_pass = all_pass and crop_ok and pipe_ok

        crop_mark = "[+] PASS" if crop_ok else "[-] FAIL"
        pipe_mark = "[+] PASS" if pipe_ok else "[-] FAIL"
        print(f"  {label:<20} target >{target:>3} FPS  |  "
              f"crop: {crop_fps:>6.0f} FPS {crop_mark}  |  "
              f"pipeline: {pipe_fps:>6.0f} FPS {pipe_mark}")

    print()
    if all_pass:
        print("  Result: ALL BENCHMARKS PASSED")
    else:
        print("  Result: SOME BENCHMARKS FAILED - see details above")

    print("=" * 88)

    if args.save:
        sys.stdout = sys.__stdout__
        with open(args.save, 'w') as f:
            f.write(''.join(output_lines))
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
