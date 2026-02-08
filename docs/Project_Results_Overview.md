# Project Results Overview

A detailed walkthrough of the Robotics Vision Pipeline: how it works end-to-end, why the architecture was chosen, how edge cases are handled, and what the benchmarks tell us.

---

## Table of Contents

1. [Application Workflow](#application-workflow)
2. [Architecture Decisions](#architecture-decisions)
3. [Rotated Crop Algorithm](#rotated-crop-algorithm)
4. [Edge-Case Clamping Algorithm](#edge-case-clamping-algorithm)
5. [Benchmark Results](#benchmark-results)
6. [Application Screenshots](#application-screenshots)

---

## Application Workflow

The system runs as two independent processes communicating over the network:

```
  video_acquisition.py                              video_crop.py
 ┌────────────────────────────┐                   ┌───────────────────────────────────────┐
 │                            │     ZeroMQ        │                                       │
 │  Webcam ──► Serialize ──►  │ ── tcp://5555 ──► │  ──► Deserialize ──► Crop ──► Display │
 │             (20B header    │     PUB/SUB       │      (frombuffer     (warp    (imshow)│
 │              + raw pixels) │                   │       + reshape)     Persp.)          │
 └────────────────────────────┘                   └───────────────────────────────────────┘
```

### Step-by-step flow

1. **Capture** - `video_acquisition.py` opens the webcam via OpenCV's `VideoCapture` and reads frames in a loop.
2. **Serialize** - Each frame is packed into a binary message: an 8-byte `double` timestamp, followed by three 4-byte `int` values (height, width, channels), followed by the raw pixel bytes. Total header overhead: 20 bytes.
3. **Transmit** - The message is published on a ZeroMQ `PUB` socket bound to `tcp://*:5555`.
4. **Receive** - `video_crop.py` connects a ZeroMQ `SUB` socket to the publisher and receives frames.
5. **Deserialize** - The header is unpacked with `struct.unpack`, and the pixel data is reshaped into a NumPy array via `np.frombuffer` (near zero-copy).
6. **Crop** - The `RotatedCropper` class applies a cached perspective transform to extract a rotated rectangular region. Config is loaded from a JSON file.
7. **Display** - The cropped result is shown with `cv2.imshow`. Press `r` to hot-reload the config, `q` to quit.

### Key source files

| File | Role |
|------|------|
| `video_acquisition.py` | Webcam capture, frame serialization, ZeroMQ publisher |
| `video_crop.py` | ZeroMQ subscriber, deserialization, crop application, display |
| `utils/rotated_crop.py` | Core algorithm: `rotated_crop()`, `RotatedCropper`, `clamp_config()`, `validate_config()` |
| `configs/rcrop_parameters.json` | Default crop parameters (normalized coordinates) |
| `configs/cropconfig.htm` | Interactive browser tool for creating/adjusting crop configs |

---

## Architecture Decisions

### 1. Network Protocol: ZeroMQ PUB/SUB

**Chosen:** ZeroMQ with PUB/SUB pattern.

| Alternative | Why it was rejected |
|-------------|-------------------|
| **Raw TCP sockets** | Requires implementing message framing (length-prefix protocol), reconnection logic, and buffering manually. Error-prone for a real-time stream. |
| **UDP** | Lowest latency, but no message framing, no guaranteed delivery, and manual fragmentation needed for frames larger than ~65 KB (every frame exceeds this). |
| **HTTP / MJPEG streaming** | High overhead per frame (HTTP headers), requires JPEG compression (lossy), and adds latency from the request-response cycle. |

**Why ZeroMQ wins:**
- **Automatic message framing** - no need for length-prefix protocols; one `send()` = one complete frame.
- **Built-in reconnection** - handles network interruptions without application-level code.
- **Frame dropping under load** - when the subscriber falls behind, old frames are silently dropped. This is the correct behavior for real-time video (you always want the latest frame).
- **Minimal API surface** - a PUB/SUB pair is ~10 lines of setup code.
- **Trade-off accepted:** one extra dependency (`pyzmq`), and slightly higher latency than raw UDP.

### 2. Serialization: Custom Binary Format

**Chosen:** `struct.pack('diii', timestamp, h, w, c) + frame.tobytes()`

| Alternative | Why it was rejected |
|-------------|-------------------|
| **pickle** | Variable-size header, security risk (arbitrary code execution on deserialization), slower for large arrays. |
| **JSON + base64** | Massive overhead: base64 encoding inflates data by ~33%, plus JSON parsing cost. |
| **JPEG compression** | Lossy - unacceptable for a vision pipeline where downstream algorithms need exact pixel values. Also adds encode/decode latency. |
| **Protocol Buffers / MessagePack** | Extra dependencies and schema management for no real benefit over a fixed 20-byte header. |

**Why custom binary wins:**
- **20 bytes of overhead** regardless of frame size (vs. pickle's variable overhead).
- **Zero-copy on deserialize** - `np.frombuffer` wraps the raw bytes without copying.
- **Cross-platform** - `struct.pack` with explicit format string works identically everywhere.
- **Trivially debuggable** - fixed header layout makes hex inspection straightforward.

### 3. Rotated Crop: Single Perspective Transform

**Chosen:** `cv2.getPerspectiveTransform` + `cv2.warpPerspective` in one step.

| Alternative | Why it was rejected |
|-------------|-------------------|
| **`cv2.warpAffine` (full rotation) + array slice** | Rotates the entire image first (expensive for large frames), then crops. Wastes computation on pixels outside the crop region. |
| **Manual NumPy coordinate remapping** | Correct but slow - per-pixel Python loops or complex indexing, no GPU/SIMD acceleration. |
| **`cv2.getRotationMatrix2D` + `cv2.getRectSubPix`** | Two separate operations; `getRectSubPix` doesn't support rotation, so the rotation must still process the full image. |

**Why perspective transform wins:**
- **Single operation** - maps directly from the rotated source rectangle to an axis-aligned output. Only the pixels inside the crop region are computed.
- **Hardware-accelerated** - OpenCV's `warpPerspective` uses SIMD (AVX2/AVX-512) and optionally CUDA.
- **Exact mapping** - perspective transform handles arbitrary quadrilateral-to-rectangle mapping with sub-pixel interpolation.

### 4. Transform Caching

The `RotatedCropper` class caches the 3x3 perspective matrix and recomputes it only when:
- The image resolution changes, or
- Any crop parameter changes (alpha, ox, oy, width, height).

For continuous video with fixed crop settings, this means the matrix is computed once and reused for every frame. The benchmarks show this yields a **~29% speedup at SD** (1719 vs 1333 FPS) and **~23% at HD** (1088 vs 881 FPS) compared to recomputing each call.

---

## Rotated Crop Algorithm

The core function `compute_rotated_corners()` calculates the four vertices of a rotated rectangle:

```
1. Start with an axis-aligned rectangle centered at (cx, cy):
       (-w/2, -h/2)  ───  (w/2, -h/2)
            │                    │
       (-w/2, h/2)   ───  (w/2, h/2)

2. Apply 2D rotation matrix (angle α, negated for image coordinates):
       ┌ cos(-α)  -sin(-α)        ┐
       │ sin(-α)   cos(-α)        │
       └                          ┘

3. Translate corners back to center (cx, cy).

4. Use these 4 source corners + 4 destination corners (axis-aligned output)
   to compute a perspective transform matrix via cv2.getPerspectiveTransform.

5. Apply cv2.warpPerspective with BORDER_CONSTANT (black fill for out-of-bounds).
```

The result is the content inside the rotated rectangle, straightened into an axis-aligned output image.

---

## Edge-Case Clamping Algorithm

When the crop rectangle partially or fully extends outside the image, the `clamp_config()` function brings it back within bounds through a multi-step process:

### Algorithm steps

```
Input: config (alpha, ox, oy, width, height), image dimensions

Step 1 ─ Normalize angle to [0, 360)
Step 2 ─ Take absolute value of negative width/height
Step 3 ─ Clamp center (ox, oy) to [0, 1]
Step 4 ─ Convert normalized coords to pixel coords
Step 5 ─ Compute the 4 rotated corners
Step 6 ─ Find the axis-aligned bounding box (min_x, max_x, min_y, max_y)
Step 7 ─ If any corner is outside image bounds:
           a. Compute scale = min(img_width / bbox_width, img_height / bbox_height)
           b. If scale < 1: shrink width and height by scale factor
           c. Recompute corners after scaling
           d. Shift center to push any remaining overflow back inside
Step 8 ─ Convert back to normalized coordinates
```

The key insight is that **scaling comes before shifting**. If the rotated bounding box is simply too large to fit at any position, it must be scaled down first. Only after the size fits do we nudge the center position to eliminate any remaining overlap with the boundary.

### Visual demonstration

The image below shows six scenarios and how the clamping algorithm handles each. For each case, the **original** crop rectangle is shown alongside the **clamped** result:

![Bounding box clamping visualization - six edge cases showing how out-of-bounds, oversized, negative, and extreme-angle crops are corrected](../img/clamping_edge_cases_visual.png)

**Cases shown (left to right, top to bottom):**

| # | Case | What happens |
|---|------|-------------|
| 1 | **Normal** | Crop fits within the image. No clamping needed. |
| 2 | **Negative width/height** | Dimensions are flipped to their absolute values; crop proceeds normally. |
| 3 | **Corner out of bounds** | Center is shifted inward so all four rotated corners land inside the image. |
| 4 | **Oversized** | Crop is larger than the image. Scaled down to the largest size that fits, then centered. |
| 5 | **Fully outside image** | Center is clamped to [0,1], then the crop is scaled and shifted to fit. |
| 6 | **Huge angle** | Angle is normalized to [0, 360). The rotated bounding box is then scaled/shifted as needed. |

In all cases, any region of the final crop that still falls outside the image after clamping is filled with black (`borderValue=0`), as specified in the assignment.

---

## Benchmark Results

Benchmarks were run with the following configuration:

| Parameter | Value |
|-----------|-------|
| Python | 3.12.4 (CPython) |
| Platform | Windows 10.0.26200, AMD64 |
| CPU cores | 8 |
| NumPy | 2.4.2 |
| OpenCV | 4.13.0 |
| SIMD | AVX-512 available |
| Methodology | 20 warm-up iterations, 7 timed rounds, ~0.5s per round, GC disabled |
| Crop config | alpha=30, size=0.4x0.4 |

### 1. Rotated Crop (cached transform)

The primary operation: applying `warpPerspective` with a pre-computed matrix.

| Resolution | Median | Mean | Std | P5 | P95 | FPS | Target | Status |
|-----------|--------|------|-----|-----|-----|-----|--------|--------|
| SD 640x480 | 581.8 us | 582.8 us | 7.3 us | 573.6 us | 593.8 us | 1719 | 60 | PASS |
| HD 1280x720 | 919.5 us | 941.0 us | 53.2 us | 889.0 us | 1.03 ms | 1088 | 45 | PASS |
| FHD 1920x1080 | 1.40 ms | 1.39 ms | 15.4 us | 1.36 ms | 1.41 ms | 716 | 30 | PASS |

**Interpretation:** The cached crop alone runs at **28x the target** at SD and **24x the target** at FHD. The low standard deviation (7.3 us at SD, 15.4 us at FHD) indicates highly consistent performance with minimal jitter. The HD resolution shows a slightly higher std (53.2 us) and a wider P5-P95 spread, likely due to cache-line effects at that intermediate buffer size, but this is negligible in practice.

### 2. Rotated Crop (recompute transform each call)

Same crop operation, but recomputing the perspective matrix every frame.

| Resolution | Median | Mean | Std | P5 | P95 | FPS |
|-----------|--------|------|-----|-----|-----|-----|
| SD 640x480 | 750.3 us | 758.8 us | 11.4 us | 748.2 us | 775.3 us | 1333 |
| HD 1280x720 | 1.13 ms | 1.14 ms | 5.7 us | 1.13 ms | 1.15 ms | 881 |
| FHD 1920x1080 | 1.58 ms | 1.58 ms | 24.9 us | 1.55 ms | 1.62 ms | 634 |

**Interpretation:** Recomputing the transform matrix adds ~170 us at SD and ~180 us at FHD per call. This confirms the caching strategy provides a measurable benefit (~22-29% throughput improvement), though even without caching the system far exceeds targets. The consistently low std values show that the matrix computation itself is deterministic.

### 3. Frame Serialization

Packing a frame into the binary wire format (`struct.pack` + `tobytes`).

| Resolution | Median | Mean | Std | P5 | P95 | FPS |
|-----------|--------|------|-----|-----|-----|-----|
| SD 640x480 | 457.7 us | 461.4 us | 8.2 us | 453.2 us | 474.4 us | 2185 |
| HD 1280x720 | 1.07 ms | 1.08 ms | 21.6 us | 1.06 ms | 1.11 ms | 936 |
| FHD 1920x1080 | 2.39 ms | 2.39 ms | 6.4 us | 2.38 ms | 2.40 ms | 418 |

**Interpretation:** Serialization is dominated by `.tobytes()`, which must copy the full pixel buffer. At FHD this means copying ~6.2 MB per frame. The linear scaling with resolution (2185 -> 936 -> 418 FPS tracks the 1x -> 3x -> 6.75x pixel count ratio) confirms this is a memory-bandwidth-bound operation. Even so, 418 FPS at FHD is well above any real-time requirement.

### 4. Frame Deserialization

Unpacking the binary message back into a NumPy array.

| Resolution | Median | Mean | Std | P5 | P95 | FPS |
|-----------|--------|------|-----|-----|-----|-----|
| SD 640x480 | 29.1 us | 29.1 us | 245 ns | 28.8 us | 29.4 us | 34349 |
| HD 1280x720 | 477.6 us | 479.3 us | 3.5 us | 476.7 us | 485.3 us | 2094 |
| FHD 1920x1080 | 1.13 ms | 1.13 ms | 4.2 us | 1.12 ms | 1.13 ms | 889 |

**Interpretation:** Deserialization is dramatically faster than serialization because `np.frombuffer` wraps the existing bytes without a full copy (the reshape is a metadata-only operation for contiguous data). At SD, it runs at **34,349 FPS** - effectively free. The jump from SD to HD (29 us to 478 us) suggests that at larger sizes, memory allocation or page faults become the bottleneck, but this remains negligible compared to the crop operation.

### 5. Full Pipeline (deserialize + crop)

The end-to-end latency a subscriber experiences per frame.

| Resolution | Median | Mean | Std | P5 | P95 | FPS | Target | Status |
|-----------|--------|------|-----|-----|-----|-----|--------|--------|
| SD 640x480 | 874.5 us | 894.4 us | 65.7 us | 813.8 us | 984.3 us | 1143 | 60 | PASS |
| HD 1280x720 | 1.67 ms | 1.68 ms | 13.3 us | 1.66 ms | 1.70 ms | 599 | 45 | PASS |
| FHD 1920x1080 | 2.86 ms | 2.87 ms | 36.3 us | 2.82 ms | 2.92 ms | 350 | 30 | PASS |

**Interpretation:** The full pipeline combines deserialization and cached crop. At every resolution, throughput exceeds the target by over an order of magnitude:

| Resolution | Target | Achieved | Headroom |
|-----------|--------|----------|----------|
| SD 640x480 | 60 FPS | 1143 FPS | **19x** |
| HD 1280x720 | 45 FPS | 599 FPS | **13x** |
| FHD 1920x1080 | 30 FPS | 350 FPS | **11x** |

This headroom means the pipeline can comfortably run on lower-spec hardware, absorb occasional latency spikes, and leave CPU budget available for other robotics tasks running on the same machine.

### Summary

```
  SD  640x480          target > 60 FPS  |  crop:   1719 FPS  PASS  |  pipeline:   1143 FPS  PASS
  HD  1280x720         target > 45 FPS  |  crop:   1088 FPS  PASS  |  pipeline:    599 FPS  PASS
  FHD 1920x1080        target > 30 FPS  |  crop:    716 FPS  PASS  |  pipeline:    350 FPS  PASS

  Result: ALL BENCHMARKS PASSED
```

---

## Application Screenshots

### Publisher window (video_acquisition.py)

![Publisher - webcam capture](../img/publisher_screenshot.png)

### Subscriber window (video_crop.py) showing the cropped output

![Subscriber - rotated crop output](../img/subscriber_screenshot.png)

### Subscriber window (video_crop.py) and Video Preview (tests/video_preview.py) windows side by side

![Full system running](../img/full_system_screenshot.png)
