"""
Rotated crop algorithm using perspective transform.

This module provides functions to extract a rotated rectangular region
from an image and map it to an axis-aligned output.
"""

import math

import numpy as np
import cv2


def validate_config(config: dict) -> list[str]:
    """
    Validate that a config dict describes a valid bounding box.

    Args:
        config: Dict with keys: alpha, ox, oy, width, height

    Returns:
        List of warning strings (empty if all is valid).

    Raises:
        ValueError: If the config is fundamentally invalid (not a box).
    """
    warnings = []

    for key in ('width', 'height'):
        val = config.get(key)
        if val is None:
            raise ValueError(f"Missing required key: {key}")
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"{key} must be a finite number, got {val!r}")
        if val == 0:
            raise ValueError(f"{key} must be non-zero, got {val}")
        if val < 0:
            warnings.append(f"{key}={val} is negative, will use absolute value")

    alpha = config.get('alpha', 0)
    if not isinstance(alpha, (int, float)) or math.isnan(alpha) or math.isinf(alpha):
        raise ValueError(f"alpha must be a finite number, got {alpha!r}")

    for key in ('ox', 'oy'):
        val = config.get(key)
        if val is not None:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{key} must be a finite number, got {val!r}")
            if val < 0 or val > 1:
                warnings.append(f"{key}={val} is outside [0, 1]")

    return warnings


def clamp_config(config: dict, img_width: int, img_height: int) -> tuple[dict, bool, list[str]]:
    """
    Clamp a config so the rotated bounding box fits within the image.

    Args:
        config: Dict with keys: alpha, ox, oy, width, height (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (clamped_config, was_clamped, warnings).
    """
    result = config.copy()
    warnings = []
    was_clamped = False

    # 1. Normalize angle to [0, 360)
    alpha = result.get('alpha', 0)
    norm_alpha = alpha % 360
    if norm_alpha != alpha:
        warnings.append(f"Angle normalized from {alpha} to {norm_alpha}")
        was_clamped = True
    result['alpha'] = norm_alpha

    # 2. Take abs of width/height if negative
    for key in ('width', 'height'):
        val = result.get(key, 0.5)
        if val < 0:
            warnings.append(f"{key} was negative ({val}), using absolute value")
            result[key] = abs(val)
            was_clamped = True

    # 3. Clamp center to [0, 1]
    for key in ('ox', 'oy'):
        val = result.get(key, 0.5)
        clamped_val = max(0.0, min(1.0, val))
        if clamped_val != val:
            warnings.append(f"{key} clamped from {val} to {clamped_val}")
            result[key] = clamped_val
            was_clamped = True

    # 4. Convert to pixels
    ox = result.get('ox', 0.5)
    oy = result.get('oy', 0.5)
    w = result.get('width', 0.5)
    h = result.get('height', 0.5)

    cx = ox * img_width
    cy = oy * img_height
    pw = w * img_width
    ph = h * img_height

    # 5. Compute rotated corners and check bounds
    corners = compute_rotated_corners(cx, cy, pw, ph, result['alpha'])
    min_x = corners[:, 0].min()
    max_x = corners[:, 0].max()
    min_y = corners[:, 1].min()
    max_y = corners[:, 1].max()

    # 6-7. Check if any corner exceeds image bounds
    if min_x < 0 or max_x > img_width or min_y < 0 or max_y > img_height:
        # 8. Compute scale factor
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        scale = 1.0
        if bbox_w > 0:
            scale = min(scale, img_width / bbox_w)
        if bbox_h > 0:
            scale = min(scale, img_height / bbox_h)

        if scale < 1.0:
            # 9. Apply scale
            pw *= scale
            ph *= scale
            warnings.append(
                f"Scaled down by {scale:.3f} to fit within image bounds"
            )
            was_clamped = True

            # 10. Recompute corners
            corners = compute_rotated_corners(cx, cy, pw, ph, result['alpha'])
            min_x = corners[:, 0].min()
            max_x = corners[:, 0].max()
            min_y = corners[:, 1].min()
            max_y = corners[:, 1].max()

        # 11. Shift center if still out of bounds
        shift_x = 0.0
        shift_y = 0.0
        if min_x < 0:
            shift_x = -min_x
        elif max_x > img_width:
            shift_x = img_width - max_x
        if min_y < 0:
            shift_y = -min_y
        elif max_y > img_height:
            shift_y = img_height - max_y

        if shift_x != 0 or shift_y != 0:
            cx += shift_x
            cy += shift_y
            warnings.append(
                f"Center shifted by ({shift_x:.1f}, {shift_y:.1f}) px to fit"
            )
            was_clamped = True

        # 12. Convert back to normalized coords
        result['ox'] = cx / img_width
        result['oy'] = cy / img_height
        result['width'] = pw / img_width
        result['height'] = ph / img_height

    return result, was_clamped, warnings


def compute_rotated_corners(center_x: float, center_y: float,
                            width: float, height: float,
                            angle_degrees: float) -> np.ndarray:
    """
    Compute the 4 corners of a rotated rectangle.

    Args:
        center_x: X coordinate of rectangle center
        center_y: Y coordinate of rectangle center
        width: Width of the rectangle
        height: Height of the rectangle
        angle_degrees: Rotation angle in degrees (counter-clockwise)

    Returns:
        4x2 numpy array of corner coordinates in order:
        [top-left, top-right, bottom-right, bottom-left]
    """
    # Convert to radians and negate because image Y-axis points downward
    angle_rad = -np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Half dimensions
    hw = width / 2
    hh = height / 2

    # Corners relative to center (before rotation)
    # Order: top-left, top-right, bottom-right, bottom-left
    corners_rel = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ], dtype=np.float32)

    # Rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float32)

    # Rotate and translate to center
    corners = corners_rel @ rotation_matrix.T + np.array([center_x, center_y])

    return corners.astype(np.float32)


def rotated_crop(image: np.ndarray, center_x: float, center_y: float,
                 width: float, height: float, angle_degrees: float) -> np.ndarray:
    """
    Extract a rotated rectangular region from an image.

    Uses perspective transform to map the rotated source region directly
    to an axis-aligned output. Out-of-bounds regions are filled with black.

    Args:
        image: Input image (numpy array, HxWxC or HxW)
        center_x: X coordinate of crop center (in pixels)
        center_y: Y coordinate of crop center (in pixels)
        width: Width of the crop region (in pixels)
        height: Height of the crop region (in pixels)
        angle_degrees: Rotation angle in degrees (counter-clockwise)

    Returns:
        Cropped image of shape (height, width, C) or (height, width)
    """
    # Compute source corners (rotated rectangle in source image)
    src_corners = compute_rotated_corners(center_x, center_y,
                                          width, height, angle_degrees)

    # Destination corners (axis-aligned rectangle)
    out_width = int(round(width))
    out_height = int(round(height))

    dst_corners = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Apply transform with black border fill
    result = cv2.warpPerspective(
        image,
        transform_matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return result


class RotatedCropper:
    """
    Optimized rotated cropper that caches transform matrix.

    Recomputes the transform only when image size or crop parameters change.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the cropper.

        Args:
            config: Optional dict with keys: alpha, ox, oy, width, height
                    All position/size values are normalized (0-1)
        """
        self._config = config or {}
        self._cached_matrix = None
        self._cached_image_size = None
        self._cached_output_size = None
        self._cached_config_hash = None
        self._was_clamped = False
        self._warnings: list[str] = []

        if config:
            validate_config(config)

    def load_config(self, config: dict):
        """Load crop configuration from dictionary.

        Raises:
            ValueError: If the config is fundamentally invalid.
        """
        validate_config(config)
        self._config = config.copy()
        self._invalidate_cache()

    def load_config_file(self, filepath: str):
        """Load crop configuration from JSON file.

        Raises:
            ValueError: If the config is fundamentally invalid.
        """
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        validate_config(config)
        self._config = config
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Clear cached transform matrix."""
        self._cached_matrix = None
        self._cached_image_size = None
        self._cached_output_size = None

    def _config_hash(self) -> tuple:
        """Generate hash of current config for cache validation."""
        return (
            self._config.get('alpha', 0),
            self._config.get('ox', 0.5),
            self._config.get('oy', 0.5),
            self._config.get('width', 0.5),
            self._config.get('height', 0.5)
        )

    def _compute_transform(self, img_height: int, img_width: int):
        """Compute and cache the perspective transform matrix."""
        # Clamp config to fit within image bounds
        clamped, self._was_clamped, self._warnings = clamp_config(
            self._config, img_width, img_height
        )

        alpha = clamped.get('alpha', 0)
        ox = clamped.get('ox', 0.5)
        oy = clamped.get('oy', 0.5)
        crop_w = clamped.get('width', 0.5)
        crop_h = clamped.get('height', 0.5)

        # Convert normalized coords to pixels
        center_x = ox * img_width
        center_y = oy * img_height
        width = crop_w * img_width
        height = crop_h * img_height

        # Compute source corners
        src_corners = compute_rotated_corners(center_x, center_y,
                                              width, height, alpha)

        # Output dimensions
        out_width = int(round(width))
        out_height = int(round(height))

        # Destination corners
        dst_corners = np.array([
            [0, 0],
            [out_width - 1, 0],
            [out_width - 1, out_height - 1],
            [0, out_height - 1]
        ], dtype=np.float32)

        self._cached_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        self._cached_image_size = (img_height, img_width)
        self._cached_output_size = (out_width, out_height)
        self._cached_config_hash = self._config_hash()

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop the image using cached transform.

        Args:
            image: Input image (HxWxC or HxW)

        Returns:
            Cropped image
        """
        img_h, img_w = image.shape[:2]
        current_size = (img_h, img_w)
        current_hash = self._config_hash()

        # Recompute transform if size or config changed
        if (self._cached_matrix is None or
            self._cached_image_size != current_size or
            self._cached_config_hash != current_hash):
            self._compute_transform(img_h, img_w)

        return cv2.warpPerspective(
            image,
            self._cached_matrix,
            self._cached_output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    @property
    def config(self) -> dict:
        """Get current configuration."""
        return self._config.copy()

    @property
    def was_clamped(self) -> bool:
        """True if the last transform required clamping."""
        return self._was_clamped

    @property
    def warnings(self) -> list[str]:
        """Human-readable list of adjustments made during clamping."""
        return self._warnings.copy()
