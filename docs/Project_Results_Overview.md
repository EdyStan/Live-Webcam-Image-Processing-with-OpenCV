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
 ┌────────────────────────────┐                   ┌───────────────────────────────────────────────┐
 │                            │     ZeroMQ        │                                               │
 │  Webcam --> Serialize -->  │ -- tcp://5555 --> │  --> Deserialize --> Clamp + Crop --> Display │
 │             (20B header    │     PUB/SUB       │      (frombuffer     (warp            (imshow)│
 │              + raw pixels) │                   │       + reshape)       Persp.)                │
 └────────────────────────────┘                   └───────────────────────────────────────────────┘
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
| `utils/rotated_crop.py` | Core algorithm: `RotatedCropper`, `clamp_config()`, `compute_rotated_corners()` |
| `configs/rcrop_parameters.json` | Default crop parameters (normalized coordinates) |
| `configs/cropconfig.htm` | Interactive browser tool for creating/adjusting crop configs |

---

## Architecture Decisions

### 1. Network Protocol: ZeroMQ PUB/SUB

Provides automatic message framing, built-in reconnection, and frame dropping under load (always showing the latest frame). Alternatives like raw TCP require manual framing/reconnection, UDP needs manual fragmentation for frames, and HTTP/MJPEG adds per-frame overhead and lossy compression.

### 2. Rotated Crop: Single Perspective Transform

`cv2.getPerspectiveTransform` + `cv2.warpPerspective` maps directly from the rotated source rectangle to an axis-aligned output in one SIMD-accelerated operation. Alternatives like `warpAffine` + slice waste computation on the full image, and manual NumPy remapping lacks hardware acceleration.

---

## Rotated Crop Algorithm

The core function `compute_rotated_corners()` calculates the four vertices of a rotated rectangle:

```
1. Start with an axis-aligned rectangle centered at (cx, cy):
       (-w/2, -h/2)  ───  (w/2, -h/2)
            │                    │
       (-w/2, h/2)   ───  (w/2, h/2)

2. Apply 2D rotation matrix (angle α, negated for image coordinates):
       
        ┌                    ┐
        │ cos(-α)   -sin(-α) │
        │ sin(-α)    cos(-α) │
        └                    ┘

3. Translate corners back to center (cx, cy).

4. Use these 4 source corners + 4 destination corners (axis-aligned output)
   to compute a perspective transform matrix via cv2.getPerspectiveTransform.

5. Apply cv2.warpPerspective to extract the crop.
```

The result is the content inside the rotated rectangle, straightened into an axis-aligned output image.

---

## Edge-Case Clamping Algorithm

When the crop rectangle partially or fully extends outside the image, the `clamp_config()` function brings it back within bounds using **asymmetric shrinking** - an approach that preserves in-bounds corners and only trims the sides that overflow, rather than uniformly scaling or translating the entire rectangle.

### Core idea: local-frame shrinking with corner anchoring

The algorithm works in the rectangle's **local coordinate frame** defined by the rotation angle. Given the two orthonormal basis vectors:

```
u = ( cos θ,  sin θ)      (along the rectangle's width)
v = (-sin θ,  cos θ)      (along the rectangle's height)

where θ = -radians(alpha)
```

the rotated rectangle is parameterized by four bounds `[u_min, u_max, v_min, v_max]`, initially `[-w/2, w/2, -h/2, h/2]`. Each corner is a combination of one u-bound and one v-bound:

```
Corner 0 (TL) = (u_min, v_min)     Corner 1 (TR) = (u_max, v_min)
Corner 3 (BL) = (u_min, v_max)     Corner 2 (BR) = (u_max, v_max)
```

A corner's global position is recovered as `(cx + cos_a * u - sin_a * v,  cy + sin_a * u + cos_a * v)`, which is **linear** in the bound variables. This linearity is what makes the problem tractable: the image containment constraints `0 <= x <= W` and `0 <= y <= H` for each corner become **linear inequalities** in the free bound variables.

### Algorithm steps

```
Input: config (alpha, ox, oy, width, height), image dimensions (W, H)

Step 1 ─ Normalize angle to [0, 360); take abs of negative width/height
Step 2 ─ Convert to pixel coords; compute the 4 rotated corners
Step 3 ─ If bbox is completely outside the image -> return None
Step 4 ─ If all corners already in bounds -> return unchanged
Step 5 ─ Asymmetric shrink (see below)
Step 6 ─ Convert the new local bounds back to (ox, oy, width, height)
```

### Step 5: asymmetric shrink solver

The solver finds new bounds `[u_min', u_max', v_min', v_max']` that **maximize the output area** `(u_max' - u_min') * (v_max' - v_min')` subject to:

- **Image containment**: all four corners lie within `[0, W] x [0, H]` (linear in the bounds).
- **Shrink-only**: bounds may only move inward (`u_min' >= u_min`, `u_max' <= u_max`, etc.).
- **Non-degeneracy**: minimum width and height of 1e-3 px.

The solver tries **anchor strategies** in priority order, preferring to keep in-bounds corners fixed:

| Priority | Strategy | Fixed bounds | Free vars | Solver |
|----------|----------|-------------|-----------|--------|
| 2 (highest) | Two adjacent in-bounds corners | 3 | 1 | Interval intersection |
| 1 | One in-bounds corner | 2 | 2 | Half-plane vertex enumeration |
| 0 (fallback) | No anchor: per-dimension shrink | 2 | 2 | Half-plane vertex enumeration |
| 0 (fallback) | No anchor: uniform scale | 0 | 1 (scale) | Interval intersection |

Within each priority level, the candidate with the largest area wins.

**1-free-variable case (interval intersection):** When two adjacent corners are anchored, three of the four bounds are fixed and one is free. The image constraints reduce to `a*t <= b` inequalities on the single free variable `t`. The feasible range is computed by intersecting these half-lines, then the area-maximizing extreme is selected. This runs in O(n) time where n is the number of constraints.

**2-free-variable case (vertex enumeration):** When one corner is anchored, two bounds are free. The constraints form a convex polygon in 2D (intersection of half-planes). Since the area objective is bilinear in the two free variables, its maximum over a convex polytope occurs at a vertex - a standard result in optimization. The solver enumerates all pairwise intersections of constraint boundaries, filters to feasible vertices, and evaluates the area at each. With ~20 constraints this is efficient (a few hundred intersection tests).

**Uniform scale fallback:** When no corners are in bounds (e.g. an oversized rectangle centered in the image), the solver finds the largest scale factor `s ∈ (0, 1]` such that all corners of the scaled rectangle fit. This reduces to interval intersection on `s`.

If the bbox is **completely outside** the image (all corners beyond one edge), `clamp_config()` returns `None`. Callers handle this gracefully: `RotatedCropper.crop()` returns `None` (and `video_crop.py` skips the frame), while `visualize_clamping.py` renders an "OUT OF BOUNDS" placeholder.

### Mathematical background

The approach combines several standard techniques from computational geometry and optimization:

- **Local coordinate frame decomposition** - representing a rotated rectangle by its local basis vectors and scalar bounds is standard in robotics and computer graphics (see Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005, Ch. 4).
- **Half-plane intersection** - the feasible region for each subproblem is the intersection of half-planes, a classical computational geometry primitive (de Berg et al., *Computational Geometry: Algorithms and Applications*, 3rd ed., Springer, 2008, Ch. 4; see also Apostolakis, Stan E. et al., "[Surface-based GPU-friendly geometry modeling for detector simulation](https://doi.org/10.1051/epjconf/202429503039)," *EPJ Web of Conferences*, vol. 295, 03039, 2024, for a modern application of half-plane intersection in geometry processing).
- **Bilinear optimization over polytopes** - the area objective `(u_max - u_min) * (v_max - v_min)` is bilinear in the two free variables, so its maximum over a convex polygon (the feasible region) is attained at a vertex. This is why the 2-free-variable solver only needs to enumerate vertices rather than search the interior (see Boyd & Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004, §4.7 on quasiconvex optimization and vertex optimality for linear/bilinear objectives over polytopes).

### Visual demonstration

The image below shows several scenarios and how the clamping algorithm handles each. For each case, the **original** crop rectangle is shown alongside the **clamped** result:

![Bounding box clamping visualization - edge cases showing how out-of-bounds, oversized, negative, and extreme-angle crops are corrected](../img/clamping_edge_cases_visual.png)

**Cases shown:**

| # | Case | What happens |
|---|------|-------------|
| 1 | **Normal** | Crop fits within the image. No clamping needed. |
| 2 | **Negative width/height** | Dimensions are flipped to their absolute values; crop proceeds normally. |
| 3 | **Corner out of bounds** | Out-of-bounds sides are shrunk inward; in-bounds corners are anchored. |
| 4 | **Oversized** | Crop is larger than the image. Uniformly scaled down to the largest size that fits. |
| 5 | **Fully outside image** | All corners are beyond the image boundary. `clamp_config()` returns `None` and the crop is skipped. |
| 6 | **Huge angle** | Angle is normalized to [0, 360). The rotated bounding box is then shrunk as needed. |
| 7 | **1 corner out** | One corner overflows. Adjacent in-bounds corners are anchored; the overflowing side shrinks to the image edge. |
| 8 | **2 corners out** | Two non-adjacent corners overflow. A single in-bounds corner is anchored; two sides shrink independently. |
| 9 | **3 corners out** | Only one corner is inside. It is anchored; the other three sides shrink to maximize the remaining area. |

In all cases, the rotation angle is preserved and no pure translation is applied - the center may shift only as a consequence of asymmetric shrinking.

---

## Benchmark Results


```
Python 3.12.4 | Windows AMD64 | 8 cores | NumPy 2.4.2 | OpenCV 4.13.0

Full pipeline (ZMQ PUB/SUB -> deserialize -> clamp + crop)
  warmup 1s, measure 5s per run

  In bounds        SD  640x480      906 FPS (target 60)  PASS
  In bounds        HD  1280x720      344 FPS (target 45)  PASS
  In bounds        FHD 1920x1080      189 FPS (target 30)  PASS

  1 corner out     SD  640x480     1489 FPS (target 60)  PASS
  1 corner out     HD  1280x720      379 FPS (target 45)  PASS
  1 corner out     FHD 1920x1080      188 FPS (target 30)  PASS

  2 corners out    SD  640x480     1387 FPS (target 60)  PASS
  2 corners out    HD  1280x720      354 FPS (target 45)  PASS
  2 corners out    FHD 1920x1080      190 FPS (target 30)  PASS
```

All targets comfortably exceeded (3x-24x headroom depending on resolution), confirming the pipeline is not bottlenecked by ZeroMQ transport even at FHD.

## Application Screenshots

### Publisher window (video_acquisition.py)

![Publisher - webcam capture](../img/publisher_screenshot.png)

---

### Subscriber window (video_crop.py) showing the cropped output

![Subscriber - rotated crop output](../img/subscriber_screenshot.png)

---

### Subscriber window (video_crop.py) and Video Preview (tests/video_preview.py) windows side by side

![Full system running](../img/full_system_screenshot.png)
