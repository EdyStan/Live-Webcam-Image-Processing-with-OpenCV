import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import compute_rotated_corners, clamp_config, RotatedCropper


# compute_rotated_corners

def test_corners_zero_rotation():
    corners = compute_rotated_corners(100, 100, 40, 20, 0)
    expected = np.array([[80, 90], [120, 90], [120, 110], [80, 110]], dtype=np.float32)
    np.testing.assert_array_almost_equal(corners, expected, decimal=5)


def test_corners_90_degrees():
    corners = compute_rotated_corners(100, 100, 40, 20, 90)
    expected = np.array([[90, 120], [90, 80], [110, 80], [110, 120]], dtype=np.float32)
    np.testing.assert_array_almost_equal(corners, expected, decimal=5)


# clamp_config

def test_clamp_in_bounds_unchanged():
    config = {'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3}
    result, was_clamped, warnings = clamp_config(config, 640, 480)
    assert not was_clamped
    assert warnings == []


def test_clamp_negative_dimensions():
    config = {'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': -0.3, 'height': -0.4}
    result, was_clamped, _ = clamp_config(config, 640, 480)
    assert result['width'] == 0.3
    assert result['height'] == 0.4
    assert was_clamped


def test_clamp_angle_normalized():
    config = {'alpha': 730, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3}
    result, was_clamped, _ = clamp_config(config, 640, 480)
    assert result['alpha'] == pytest.approx(10.0)
    assert was_clamped


def test_clamp_oversized_scaled_down():
    config = {'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 1.5, 'height': 1.5}
    result, was_clamped, _ = clamp_config(config, 640, 480)
    assert was_clamped
    assert result['width'] <= 1.0
    assert result['height'] <= 1.0


def test_clamp_center_out_of_range_completely():
    """Bbox completely outside image returns None."""
    config = {'alpha': 0, 'ox': 2.0, 'oy': -0.5, 'width': 0.2, 'height': 0.2}
    result, was_clamped, warnings = clamp_config(config, 640, 480)
    assert result is None
    assert was_clamped
    assert any("completely outside" in w for w in warnings)


def test_clamp_center_partially_out_of_range():
    """Bbox partially outside image gets clamped (not None)."""
    config = {'alpha': 0, 'ox': 0.9, 'oy': 0.9, 'width': 0.5, 'height': 0.5}
    result, was_clamped, _ = clamp_config(config, 640, 480)
    assert result is not None
    assert was_clamped


# RotatedCropper

def test_cropper_output_dimensions():
    cropper = RotatedCropper({'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.5, 'height': 0.5})
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = cropper.crop(img)
    assert result.shape == (240, 320, 3)


def test_cropper_config_change_updates_output():
    cropper = RotatedCropper({'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.5, 'height': 0.5})
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cropper.crop(img).shape[1] == 320

    cropper.load_config({'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.25, 'height': 0.25})
    assert cropper.crop(img).shape[1] == 160


def test_cropper_was_clamped_flag():
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    normal = RotatedCropper({'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3})
    normal.crop(img)
    assert not normal.was_clamped

    oversized = RotatedCropper({'alpha': 15, 'ox': 0.5, 'oy': 0.5, 'width': 1.5, 'height': 1.8})
    oversized.crop(img)
    assert oversized.was_clamped


def test_cropper_returns_none_when_completely_outside():
    cropper = RotatedCropper({'alpha': 0, 'ox': 3.0, 'oy': 3.0, 'width': 0.1, 'height': 0.1})
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cropper.crop(img) is None
    assert cropper.was_clamped

# clamp_config – asymmetric shrink tests


def _corners_from_result(result, img_w, img_h):
    """Helper: compute pixel corners from a clamp_config result."""
    cx = result["ox"] * img_w
    cy = result["oy"] * img_h
    pw = result["width"] * img_w
    ph = result["height"] * img_h
    return compute_rotated_corners(cx, cy, pw, ph, result["alpha"])


def _assert_corners_in_image(corners, img_w, img_h, tol=0.5):
    """Assert every corner lies within [0, W] x [0, H] (with tolerance)."""
    assert corners[:, 0].min() >= -tol
    assert corners[:, 0].max() <= img_w + tol
    assert corners[:, 1].min() >= -tol
    assert corners[:, 1].max() <= img_h + tol


def test_clamp_shrink_left_edge_alpha0():
    """Left side overflows → anchor right edge (TR,BR), shrink width only."""
    img_w, img_h = 640, 480
    # cx=224, cy=240, pw=512(hw=256), ph=192(hh=96)
    # TL=(-32,144) out, TR=(480,144) in, BR=(480,336) in, BL=(-32,336) out
    config = {"alpha": 0, "ox": 0.35, "oy": 0.50, "width": 0.80, "height": 0.40}

    result, was_clamped, _ = clamp_config(config, img_w, img_h)
    assert result is not None and was_clamped
    corners = _corners_from_result(result, img_w, img_h)

    # Anchored corners stay put; shrunk corners land on x=0
    np.testing.assert_array_almost_equal(corners[1], [480, 144], decimal=1)  # TR
    np.testing.assert_array_almost_equal(corners[2], [480, 336], decimal=1)  # BR
    np.testing.assert_array_almost_equal(corners[0], [0, 144], decimal=1)    # TL
    np.testing.assert_array_almost_equal(corners[3], [0, 336], decimal=1)    # BL

    assert result["height"] == pytest.approx(config["height"], abs=1e-6)


def test_clamp_shrink_bottom_edge_alpha0():
    """Bottom side overflows → anchor top edge (TL,TR), shrink height only."""
    img_w, img_h = 640, 480
    # cx=320, cy=384, pw=256(hw=128), ph=240(hh=120)
    # TL=(192,264) in, TR=(448,264) in, BR=(448,504) out, BL=(192,504) out
    config = {"alpha": 0, "ox": 0.50, "oy": 0.80, "width": 0.40, "height": 0.50}

    result, was_clamped, _ = clamp_config(config, img_w, img_h)
    assert result is not None and was_clamped
    corners = _corners_from_result(result, img_w, img_h)

    # Anchored top edge stays; bottom edge shrinks to y=480
    np.testing.assert_array_almost_equal(corners[0], [192, 264], decimal=1)  # TL
    np.testing.assert_array_almost_equal(corners[1], [448, 264], decimal=1)  # TR
    np.testing.assert_array_almost_equal(corners[2], [448, 480], decimal=1)  # BR
    np.testing.assert_array_almost_equal(corners[3], [192, 480], decimal=1)  # BL

    assert result["width"] == pytest.approx(config["width"], abs=1e-6)


def test_clamp_anchor_single_corner_alpha0():
    """Only BL corner is in bounds → anchor it and shrink the other three sides."""
    img_w, img_h = 640, 480
    # cx=480, cy=120, pw=512(hw=256), ph=384(hh=192)
    # TL=(224,-72) out, TR=(736,-72) out, BR=(736,312) out, BL=(224,312) in
    config = {"alpha": 0, "ox": 0.75, "oy": 0.25, "width": 0.80, "height": 0.80}

    result, was_clamped, _ = clamp_config(config, img_w, img_h)
    assert result is not None and was_clamped
    corners = _corners_from_result(result, img_w, img_h)

    # BL (corners[3]) must stay at its original position
    np.testing.assert_array_almost_equal(corners[3], [224, 312], decimal=1)
    _assert_corners_in_image(corners, img_w, img_h)
