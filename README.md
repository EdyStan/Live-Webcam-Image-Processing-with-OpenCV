# Robotics Vision System - Real-Time Video Capture & Rotated Crop Processing

A Python pipeline that captures webcam frames, transmits them over a network using ZeroMQ, and applies a configurable rotated crop transform in real time.

---

## Project Structure

```
robotics_interview_python/
│
├── video_acquisition.py        # Webcam capture & frame publisher
├── video_crop.py               # Frame subscriber & rotated crop display
├── requirements.txt            # Python dependencies
├── DESIGN_NOTES.md             # Architecture decisions & rationale
│
├── utils/
│   └── rotated_crop.py         # Core rotated crop algorithm (RotatedCropper)
│
├── configs/
│   ├── rcrop_parameters.json   # Default crop configuration
│   ├── config_corner_out.json  # Edge case: corner outside bounds
│   ├── config_inverted.json    # Edge case: negative dimensions
│   ├── config_outside.json     # Edge case: crop fully outside image
│   ├── config_oversized.json   # Edge case: crop larger than image
│   └── cropconfig.htm          # Interactive HTML tool for generating configs
│
├── tests/
│   ├── test_rotated_crop.py    # Unit tests for the crop algorithm
│   ├── test_integration.py     # Integration tests (serialization, ZMQ, pipeline)
│   ├── benchmark.py            # Performance benchmarking suite
│   ├── video_preview.py        # Video preview utility
│   └── visualize_clamping.py   # Visualization of edge-case clamping
│
├── docs/
│   ├── ASSIGNMENT.md              # Original assignment specification
│   ├── ASSIGNMENT.pdf             # PDF version
│   └── Project_Results_Overview.md # Workflow, architecture, benchmarks & screenshots
│
└── img/
    ├── edge_case.jpg              # Reference: out-of-bounds regions filled black
    ├── pattern.jpg                # Test pattern image
    ├── clamping_edge_cases_visual.png  # Clamping algorithm visualization
    ├── publisher_screenshot.png   # Publisher window screenshot
    ├── subscriber_screenshot.png  # Subscriber window screenshot
    └── full_system_screenshot.png # Full system running side by side
```

---

## How It Works

The system is split into two scripts connected over a ZeroMQ PUB/SUB socket:

```
  video_acquisition.py                       video_crop.py
 ┌──────────────────────┐    ZeroMQ       ┌──────────────────────────────┐
 │  Webcam  ──► Encode  │ ──tcp://5555──► │  Decode ──► Crop ──► Display │
 └──────────────────────┘                 └──────────────────────────────┘
```

1. **`video_acquisition.py`** opens the webcam, captures frames, serializes them into a compact binary format (20-byte header + raw pixels), and publishes them on a ZeroMQ PUB socket.
2. **`video_crop.py`** subscribes to the stream, deserializes each frame, applies a rotated crop transform (configured via JSON), and displays the result in a window.

The rotated crop algorithm in **`utils/rotated_crop.py`** uses `cv2.getPerspectiveTransform` to map a rotated rectangle directly to an axis-aligned output in a single operation. It caches the transform matrix for performance and handles edge cases (out-of-bounds regions are filled with black).

---

## Quick Start

### Prerequisites

- Python 3.8+
- A webcam (for live capture)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the publisher (webcam capture)

```bash
python video_acquisition.py
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Camera device index |
| `--width` | `640` | Capture width |
| `--height` | `480` | Capture height |
| `--fps` | `30` | Target frame rate |
| `--port` | `5555` | ZeroMQ publish port |

Example:

```bash
python video_acquisition.py --camera 0 --width 1280 --height 720 --fps 30
```

### 3. Start the subscriber (crop & display)

In a **separate terminal**:

```bash
python video_crop.py --config configs/rcrop_parameters.json
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Publisher hostname |
| `--port` | `5555` | Publisher port |
| `--config` | *(required)* | Path to crop config JSON |

### Keyboard Controls (video_crop.py)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reload crop config from disk |

---

## Crop Configuration

Crop parameters are defined in a JSON file with normalized coordinates (values between 0 and 1, relative to image dimensions):

```json
{
    "alpha": 353.34,
    "ox": 0.434,
    "oy": 0.493,
    "width": 0.241,
    "height": 0.627
}
```

| Field | Description |
|-------|-------------|
| `alpha` | Rotation angle in degrees (counter-clockwise) |
| `ox` | Crop center X (fraction of image width) |
| `oy` | Crop center Y (fraction of image height) |
| `width` | Crop width (fraction of image width) |
| `height` | Crop height (fraction of image height) |

To create or adjust configs visually, open **`configs/cropconfig.htm`** in a browser. It provides an interactive tool to drag/rotate a crop rectangle and export the JSON.

### Edge Case Handling

When the crop rectangle extends beyond the image boundaries, out-of-bounds regions are filled with black:

![Edge case - out of bounds filled with black](img/edge_case.jpg)

Several edge-case configs are provided in `configs/` for testing.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/test_rotated_crop.py -v

# Run only integration tests
pytest tests/test_integration.py -v

# Run benchmarks
python tests/benchmark.py
```

---

## Test Utilities

| Script | Purpose |
|--------|---------|
| `tests/video_preview.py` | Preview webcam feed without cropping |
| `tests/visualize_clamping.py` | Visualize how edge-case configs are clamped to valid bounds |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python >= 4.5.0` | Image capture & perspective transforms |
| `pyzmq >= 22.0.0` | Network frame transmission (PUB/SUB) |
| `numpy >= 1.20.0` | Array operations |
| `pytest >= 7.0.0` | Test framework |

---

## Further Documentation

- **[Project_Results_Overview.md](docs/Project_Results_Overview.md)** - Detailed workflow explanation, architecture decisions, clamping algorithm walkthrough, and benchmark results with interpretation.
