import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import compute_rotated_corners, rotated_crop, clamp_config, RotatedCropper


# compute_rotated_corners

def test_corners_zero_rotation():
    corners = compute_rotated_corners(100, 100, 40, 20, 0)
    expected = np.array([[80, 90], [120, 90], [120, 110], [80, 110]], dtype=np.float32)
    np.testing.assert_array_almost_equal(corners, expected, decimal=5)


def test_corners_90_degrees():
    corners = compute_rotated_corners(100, 100, 40, 20, 90)
    expected = np.array([[90, 120], [90, 80], [110, 80], [110, 120]], dtype=np.float32)
    np.testing.assert_array_almost_equal(corners, expected, decimal=5)


# rotated_crop

def test_crop_output_dimensions():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    result = rotated_crop(img, 150, 100, 80, 60, 30)
    assert result.shape == (60, 80, 3)


def test_crop_grayscale():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    result = rotated_crop(img, 50, 50, 30, 20, 0)
    assert result.shape == (20, 30)


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
