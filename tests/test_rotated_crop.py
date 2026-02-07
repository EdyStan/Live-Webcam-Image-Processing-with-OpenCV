"""
Unit tests for rotated_crop module.
"""

import numpy as np
import pytest
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import (
    compute_rotated_corners, rotated_crop, RotatedCropper,
    validate_config, clamp_config,
)


class TestComputeRotatedCorners:
    """Tests for corner computation at key angles."""

    def test_zero_rotation(self):
        """At 0 degrees, corners should be axis-aligned."""
        corners = compute_rotated_corners(100, 100, 40, 20, 0)
        expected = np.array([
            [80, 90], [120, 90], [120, 110], [80, 110]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(corners, expected, decimal=5)

    def test_90_degree_rotation(self):
        """At 90 degrees CCW, width and height swap in visual space."""
        corners = compute_rotated_corners(100, 100, 40, 20, 90)
        expected = np.array([
            [90, 120], [90, 80], [110, 80], [110, 120]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(corners, expected, decimal=5)

    def test_45_degree_rotation(self):
        """At 45 degrees, corners should be at known diamond positions."""
        corners = compute_rotated_corners(0, 0, 2 * np.sqrt(2), 2 * np.sqrt(2), 45)
        expected = np.array([
            [-2, 0], [0, -2], [2, 0], [0, 2]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(corners, expected, decimal=5)


class TestRotatedCrop:
    """Tests for the rotated_crop function."""

    def test_output_dimensions(self):
        """Output should match requested width/height."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cropped = rotated_crop(img, 150, 100, 80, 60, 30)
        assert cropped.shape == (60, 80, 3)

    def test_grayscale_image(self):
        """Should work with single-channel images."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        cropped = rotated_crop(gray, 50, 50, 30, 20, 0)
        assert cropped.shape == (20, 30)


class TestRotatedCropper:
    """Tests for the RotatedCropper class."""

    def test_crop_dimensions(self):
        """Cropper should produce correct output dimensions."""
        cropper = RotatedCropper({
            'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.5, 'height': 0.5
        })
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = cropper.crop(img)
        assert result.shape == (240, 320, 3)

    def test_cache_invalidation_on_config_change(self):
        """Changing config should produce different output size."""
        cropper = RotatedCropper({
            'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.5, 'height': 0.5
        })
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert cropper.crop(img).shape[1] == 320

        cropper.load_config({
            'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.25, 'height': 0.25
        })
        assert cropper.crop(img).shape[1] == 160

    def test_real_config(self):
        """End-to-end with actual rcrop_parameters.json values."""
        config = {
            "alpha": 353.34186649129526,
            "ox": 0.4335923492789316,
            "oy": 0.4927726531569474,
            "width": 0.24089365107766106,
            "height": 0.6271868980605269,
        }
        cropper = RotatedCropper(config)
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = cropper.crop(img)
        assert result.shape == (
            int(round(config['height'] * 480)),
            int(round(config['width'] * 640)),
            3,
        )


class TestValidateConfig:
    """Tests for validate_config - only distinct failure modes."""

    def test_valid_config(self):
        """Valid config produces no warnings."""
        assert validate_config({'alpha': 30, 'ox': 0.5, 'oy': 0.5,
                                'width': 0.3, 'height': 0.4}) == []

    def test_missing_key_raises(self):
        """Missing width or height raises ValueError."""
        with pytest.raises(ValueError, match="Missing required key"):
            validate_config({'height': 0.5})

    def test_non_finite_raises(self):
        """NaN/Inf/non-numeric values raise ValueError."""
        for bad in [float('nan'), float('inf'), 'abc']:
            with pytest.raises(ValueError):
                validate_config({'width': bad, 'height': 0.5})

    def test_zero_raises(self):
        """Zero width/height raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            validate_config({'width': 0, 'height': 0.5})

    def test_negative_warns(self):
        """Negative width/height warns but doesn't raise."""
        warnings = validate_config({'width': -0.3, 'height': 0.4})
        assert any("negative" in w for w in warnings)

    def test_out_of_range_center_warns(self):
        """ox/oy outside [0,1] produces a warning."""
        warnings = validate_config({'width': 0.3, 'height': 0.4, 'ox': 2.5})
        assert any("ox" in w for w in warnings)


class TestClampConfig:
    """Tests for clamp_config - each distinct clamping operation."""

    def test_in_bounds_unchanged(self):
        """Config already in bounds should not be clamped."""
        config = {'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3}
        _, was_clamped, warnings = clamp_config(config, 640, 480)
        assert not was_clamped
        assert warnings == []

    def test_negative_dimensions_fixed(self):
        """Negative width/height become positive."""
        config = {'alpha': 180, 'ox': 0.5, 'oy': 0.5, 'width': -0.3, 'height': -0.4}
        result, was_clamped, _ = clamp_config(config, 640, 480)
        assert result['width'] == 0.3
        assert result['height'] == 0.4
        assert was_clamped

    def test_angle_normalized(self):
        """Angle > 360 is reduced to [0, 360)."""
        config = {'alpha': 730, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3}
        result, was_clamped, _ = clamp_config(config, 640, 480)
        assert result['alpha'] == pytest.approx(10.0)
        assert was_clamped

    def test_oversized_scaled_down(self):
        """Oversized bbox is scaled to fit within image bounds."""
        config = {'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 1.5, 'height': 1.8}
        result, was_clamped, _ = clamp_config(config, 640, 480)
        assert was_clamped
        # Verify corners fit
        cx, cy = result['ox'] * 640, result['oy'] * 480
        pw, ph = result['width'] * 640, result['height'] * 480
        corners = compute_rotated_corners(cx, cy, pw, ph, result['alpha'])
        assert corners[:, 0].min() >= -0.5
        assert corners[:, 0].max() <= 640.5
        assert corners[:, 1].min() >= -0.5
        assert corners[:, 1].max() <= 480.5

    def test_out_of_bounds_center_shifted(self):
        """Center outside [0,1] is clamped and bbox shifted to fit."""
        config = {'alpha': 45, 'ox': 2.5, 'oy': -0.5, 'width': 0.3, 'height': 0.3}
        result, was_clamped, _ = clamp_config(config, 640, 480)
        assert was_clamped
        assert 0 <= result['ox'] <= 1
        assert 0 <= result['oy'] <= 1


class TestCropperClamping:
    """Tests for RotatedCropper validation/clamping integration."""

    def test_invalid_config_rejected(self):
        """NaN/zero configs raise ValueError on init and load."""
        with pytest.raises(ValueError):
            RotatedCropper({'width': float('nan'), 'height': 0.5})
        cropper = RotatedCropper()
        with pytest.raises(ValueError):
            cropper.load_config({'width': 0, 'height': 0.5})

    def test_was_clamped_flag(self):
        """was_clamped reflects whether clamping happened."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        ok = RotatedCropper({'alpha': 0, 'ox': 0.5, 'oy': 0.5,
                              'width': 0.3, 'height': 0.3})
        ok.crop(img)
        assert not ok.was_clamped

        big = RotatedCropper({'alpha': 15, 'ox': 0.5, 'oy': 0.5,
                               'width': 1.5, 'height': 1.8})
        big.crop(img)
        assert big.was_clamped
        assert len(big.warnings) > 0

    def test_edge_configs_produce_valid_output(self):
        """All edge-case configs should produce non-empty crops after clamping."""
        configs = [
            {'alpha': 25, 'ox': 0.15, 'oy': 0.2, 'width': 0.35, 'height': 0.45},
            {'alpha': 15, 'ox': 0.5, 'oy': 0.5, 'width': 1.5, 'height': 1.8},
            {'alpha': 180, 'ox': 0.8, 'oy': 0.85, 'width': -0.3, 'height': -0.4},
            {'alpha': 7432.891, 'ox': 0.123, 'oy': 0.987, 'width': 0.031, 'height': 0.271},
        ]
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for config in configs:
            result = RotatedCropper(config).crop(img)
            assert result.shape[0] > 0 and result.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
