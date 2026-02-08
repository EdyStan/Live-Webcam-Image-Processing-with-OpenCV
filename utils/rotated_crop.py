import numpy as np
import cv2

# Corner i uses bounds (u_idx, v_idx) from [u_min, u_max, v_min, v_max]
# TL=0, TR=1, BR=2, BL=3
_CORNER_BOUNDS = [(0, 2), (1, 2), (1, 3), (0, 3)]
_ADJ_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _solve_1d(orig, f, constraints, eps):
    """Solve 1-free-variable subproblem. Returns (bounds, area) or None."""
    lo = -1e18
    hi = 1e18
    for coeffs, rhs in constraints:
        a = coeffs.get(f, 0.0)
        if abs(a) < 1e-12:
            if 0 > rhs + eps:
                return None
            continue
        val = rhs / a
        if a > 0:
            hi = min(hi, val)
        else:
            lo = max(lo, val)
    if lo > hi + eps:
        return None
    # Maximize area: u_max/v_max -> maximize, u_min/v_min -> minimize
    optimal = hi if f in (1, 3) else lo
    optimal = max(lo, min(hi, optimal))
    new_bounds = list(orig)
    new_bounds[f] = optimal
    area = (new_bounds[1] - new_bounds[0]) * (new_bounds[3] - new_bounds[2])
    return (new_bounds, area) if area > eps * eps else None


def _solve_2d(orig, free_list, constraints, eps):
    """Solve 2-free-variable subproblem via vertex enumeration.
    Returns (bounds, area) or None."""
    f0, f1 = free_list
    planes = [(c.get(f0, 0.0), c.get(f1, 0.0), r) for c, r in constraints]
    n = len(planes)
    vertices = []
    for i in range(n):
        for j in range(i + 1, n):
            a0i, a1i, bi = planes[i]
            a0j, a1j, bj = planes[j]
            det = a0i * a1j - a0j * a1i
            if abs(det) < 1e-12:
                continue
            x = (bi * a1j - bj * a1i) / det
            y = (a0i * bj - a0j * bi) / det
            feasible = True
            for a0k, a1k, bk in planes:
                if a0k * x + a1k * y > bk + eps:
                    feasible = False
                    break
            if feasible:
                vertices.append((x, y))
    if not vertices:
        return None
    best_bounds = None
    best_area = -1.0
    for x, y in vertices:
        nb = list(orig)
        nb[f0] = x
        nb[f1] = y
        a = (nb[1] - nb[0]) * (nb[3] - nb[2])
        if a > best_area:
            best_area = a
            best_bounds = nb
    return (best_bounds, best_area) if best_area > eps * eps else None


def _build_constraints(orig, free_list, cx, cy, cos_a, sin_a, W, H, eps):
    """Build linear constraints for the subproblem.
    Returns list of (coeffs_dict, rhs) where sum(coeffs[f]*var_f) <= rhs."""
    constraints = []
    for ci in range(4):
        ui, vi = _CORNER_BOUNDS[ci]
        # x = cx + cos_a * bounds[ui] - sin_a * bounds[vi]
        x_const = cx
        x_coeffs = {}
        if ui in free_list:
            x_coeffs[ui] = cos_a
        else:
            x_const += cos_a * orig[ui]
        if vi in free_list:
            x_coeffs[vi] = x_coeffs.get(vi, 0.0) - sin_a
        else:
            x_const -= sin_a * orig[vi]

        # y = cy + sin_a * bounds[ui] + cos_a * bounds[vi]
        y_const = cy
        y_coeffs = {}
        if ui in free_list:
            y_coeffs[ui] = sin_a
        else:
            y_const += sin_a * orig[ui]
        if vi in free_list:
            y_coeffs[vi] = y_coeffs.get(vi, 0.0) + cos_a
        else:
            y_const += cos_a * orig[vi]

        # x >= 0 -> -coeffs <= x_const
        constraints.append(
            ({f: -x_coeffs.get(f, 0.0) for f in free_list}, x_const))
        # x <= W -> coeffs <= W - x_const
        constraints.append(
            ({f: x_coeffs.get(f, 0.0) for f in free_list}, W - x_const))
        # y >= 0
        constraints.append(
            ({f: -y_coeffs.get(f, 0.0) for f in free_list}, y_const))
        # y <= H
        constraints.append(
            ({f: y_coeffs.get(f, 0.0) for f in free_list}, H - y_const))

    # Shrink-only constraints
    for f in free_list:
        if f in (0, 2):  # u_min or v_min: can only increase
            constraints.append(({f: -1.0}, -orig[f]))
        else:  # u_max or v_max: can only decrease
            constraints.append(({f: 1.0}, orig[f]))

    # Non-degeneracy: u_max - u_min >= eps, v_max - v_min >= eps
    for lo_idx, hi_idx in [(0, 1), (2, 3)]:
        if lo_idx in free_list or hi_idx in free_list:
            coeffs = {}
            fixed_part = 0.0
            if lo_idx in free_list:
                coeffs[lo_idx] = 1.0
            else:
                fixed_part += orig[lo_idx]
            if hi_idx in free_list:
                coeffs[hi_idx] = -1.0
            else:
                fixed_part -= orig[hi_idx]
            constraints.append((coeffs, -eps - fixed_part))

    return constraints


def _solve_subproblem(orig, free_list, cx, cy, cos_a, sin_a, W, H, eps):
    """Solve the constrained shrink subproblem for given free bounds.
    Returns (new_bounds, area) or None if infeasible."""
    n_free = len(free_list)
    if n_free == 0:
        # Just verify all corners in bounds
        for ci in range(4):
            ui, vi = _CORNER_BOUNDS[ci]
            x = cx + cos_a * orig[ui] - sin_a * orig[vi]
            y = cy + sin_a * orig[ui] + cos_a * orig[vi]
            if x < -eps or x > W + eps or y < -eps or y > H + eps:
                return None
        return list(orig), (orig[1] - orig[0]) * (orig[3] - orig[2])

    constraints = _build_constraints(
        orig, free_list, cx, cy, cos_a, sin_a, W, H, eps)

    if n_free == 1:
        return _solve_1d(orig, free_list[0], constraints, eps)
    elif n_free == 2:
        return _solve_2d(orig, free_list, constraints, eps)
    return None


def _find_uniform_scale(cx, cy, hw, hh, cos_a, sin_a, W, H):
    """Find largest scale s in (0, 1] so all corners of the scaled rect
    lie within [0,W]x[0,H]."""
    s_lo = 0.0
    s_hi = 1.0
    for u_sign in (-1, 1):
        for v_sign in (-1, 1):
            ax = cos_a * (u_sign * hw) - sin_a * (v_sign * hh)
            ay = sin_a * (u_sign * hw) + cos_a * (v_sign * hh)
            for a, base, lo, hi in [(ax, cx, 0.0, W), (ay, cy, 0.0, H)]:
                if abs(a) < 1e-12:
                    if base < lo - 1e-6 or base > hi + 1e-6:
                        return None
                    continue
                v_lo_val = lo - base
                v_hi_val = hi - base
                if a > 0:
                    s_lo = max(s_lo, v_lo_val / a)
                    s_hi = min(s_hi, v_hi_val / a)
                else:
                    s_lo = max(s_lo, v_hi_val / a)
                    s_hi = min(s_hi, v_lo_val / a)
    if s_lo > s_hi + 1e-12:
        return None
    return max(s_hi, 0.0) if s_hi > 1e-12 else None


def _solve_asymmetric_shrink(cx, cy, pw, ph, alpha_deg, W, H, eps=1e-3):
    """Find the largest sub-rectangle (same angle, asymmetric shrink only)
    that fits within [0,W]x[0,H].

    Works in the rectangle's local coordinate frame:
      u = (cos_a, sin_a), v = (-sin_a, cos_a)
    where angle_rad = -radians(alpha_deg).

    Bounds: [u_min, u_max, v_min, v_max], initially [-hw, hw, -hh, hh].
    Free bounds may only shrink inward.  In-bounds corners are anchored
    (their bounds stay fixed) when possible.

    Returns (new_bounds, warnings) or (None, warnings) if infeasible.
    """
    angle_rad = -np.radians(alpha_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    hw, hh = pw / 2.0, ph / 2.0
    orig = [-hw, hw, -hh, hh]

    def _corner_in_image(bounds, ci):
        ui, vi = _CORNER_BOUNDS[ci]
        x = cx + cos_a * bounds[ui] - sin_a * bounds[vi]
        y = cy + sin_a * bounds[ui] + cos_a * bounds[vi]
        return -eps <= x <= W + eps and -eps <= y <= H + eps

    in_bounds = [_corner_in_image(orig, i) for i in range(4)]

    if all(in_bounds):
        return list(orig), []

    # Collect candidate solutions: (priority, area, bounds, warnings)
    candidates = []

    # Priority 2: anchor two adjacent in-bounds corners (1 free var)
    for i, j in _ADJ_PAIRS:
        if not (in_bounds[i] and in_bounds[j]):
            continue
        fixed = set()
        for c in (i, j):
            fixed.update(_CORNER_BOUNDS[c])
        free = sorted(set(range(4)) - fixed)
        result = _solve_subproblem(
            orig, free, cx, cy, cos_a, sin_a, W, H, eps)
        if result:
            bnd, area = result
            candidates.append((
                2, area, bnd,
                [f"Anchored corners {i},{j}; shrunk bound {free}"]))

    # Priority 1: anchor one in-bounds corner (2 free vars)
    for i in range(4):
        if not in_bounds[i]:
            continue
        fixed = set(_CORNER_BOUNDS[i])
        free = sorted(set(range(4)) - fixed)
        result = _solve_subproblem(
            orig, free, cx, cy, cos_a, sin_a, W, H, eps)
        if result:
            bnd, area = result
            candidates.append((
                1, area, bnd,
                [f"Anchored corner {i}; shrunk bounds {free}"]))

    # Priority 0: no anchor
    # 0a. Try freeing each dimension pair independently (2 free vars)
    for free in [[0, 1], [2, 3]]:
        result = _solve_subproblem(
            orig, free, cx, cy, cos_a, sin_a, W, H, eps)
        if result:
            bnd, area = result
            candidates.append((
                0, area, bnd,
                [f"Shrunk bounds {free} (no anchor)"]))

    # 0b. Uniform scale from center
    scale = _find_uniform_scale(cx, cy, hw, hh, cos_a, sin_a, W, H)
    if scale is not None and scale * max(pw, ph) > eps:
        new_hw, new_hh = hw * scale, hh * scale
        bnd = [-new_hw, new_hw, -new_hh, new_hh]
        area = (2 * new_hw) * (2 * new_hh)
        candidates.append((
            0, area, bnd,
            [f"Uniform scale {scale:.4f} (no anchor)"]))

    if not candidates:
        return None, ["No feasible rectangle found"]

    # Pick: highest priority first, then largest area
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, best_bounds, best_warnings = candidates[0]
    return best_bounds, best_warnings


def clamp_config(config, img_width, img_height):
    """Clamp a config so the rotated bounding box fits within the image.

    Uses asymmetric shrinking: in-bounds corners are anchored when possible,
    out-of-bounds sides are shrunk inward.  The center may shift as a
    *consequence* of asymmetric shrinking, but no pure translation step
    is applied.
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

    # 3. Convert to pixels
    ox = result.get('ox', 0.5)
    oy = result.get('oy', 0.5)
    w = result.get('width', 0.5)
    h = result.get('height', 0.5)

    cx = ox * img_width
    cy = oy * img_height
    pw = w * img_width
    ph = h * img_height

    corners = compute_rotated_corners(cx, cy, pw, ph, result['alpha'])

    # 4. Check if bbox is completely outside the image
    if (corners[:, 0].max() <= 0 or corners[:, 0].min() >= img_width or
            corners[:, 1].max() <= 0 or corners[:, 1].min() >= img_height):
        warnings.append("Bounding box is completely outside the image")
        return None, True, warnings

    # 5. Check if already fully in bounds
    if (corners[:, 0].min() >= -1e-3 and
            corners[:, 0].max() <= img_width + 1e-3 and
            corners[:, 1].min() >= -1e-3 and
            corners[:, 1].max() <= img_height + 1e-3):
        return result, was_clamped, warnings

    # 6. Asymmetric shrink
    new_bounds, shrink_warnings = _solve_asymmetric_shrink(
        cx, cy, pw, ph, result['alpha'], img_width, img_height)

    if new_bounds is None:
        warnings.extend(shrink_warnings)
        warnings.append("Bounding box could not be fit within the image")
        return None, True, warnings

    warnings.extend(shrink_warnings)
    was_clamped = True

    # 7. Convert local bounds back to config
    u_min, u_max, v_min, v_max = new_bounds
    new_w = u_max - u_min
    new_h = v_max - v_min

    # New center in local frame (relative to original center)
    u_center = (u_min + u_max) / 2.0
    v_center = (v_min + v_max) / 2.0

    # Map local center offset to global
    angle_rad = -np.radians(result['alpha'])
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    new_cx = cx + cos_a * u_center - sin_a * v_center
    new_cy = cy + sin_a * u_center + cos_a * v_center

    result['ox'] = new_cx / img_width
    result['oy'] = new_cy / img_height
    result['width'] = new_w / img_width
    result['height'] = new_h / img_height

    return result, was_clamped, warnings


def compute_rotated_corners(center_x, center_y, width, height, angle_degrees):
    """Compute the 4 corners of a rotated rectangle."""
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

class RotatedCropper:

    def __init__(self, config=None):
        self._config = config or {}
        self._cached_matrix = None
        self._cached_image_size = None
        self._cached_output_size = None
        self._cached_config_hash = None
        self._was_clamped = False
        self._warnings = []

    def load_config(self, config):
        """Load crop configuration from dictionary."""
        self._config = config.copy()
        self._invalidate_cache()

    def load_config_file(self, filepath):
        """Load crop configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        self._config = config
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Clear cached transform matrix."""
        self._cached_matrix = None
        self._cached_image_size = None
        self._cached_output_size = None

    def _config_hash(self):
        """Generate hash of current config for cache validation."""
        return (
            self._config.get('alpha', 0),
            self._config.get('ox', 0.5),
            self._config.get('oy', 0.5),
            self._config.get('width', 0.5),
            self._config.get('height', 0.5)
        )

    def _compute_transform(self, img_height, img_width):
        """Compute and cache the perspective transform matrix."""
        # Clamp config to fit within image bounds
        clamped, self._was_clamped, self._warnings = clamp_config(
            self._config, img_width, img_height
        )

        if clamped is None:
            self._cached_matrix = None
            self._cached_image_size = (img_height, img_width)
            self._cached_output_size = None
            self._cached_config_hash = self._config_hash()
            return

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

    def crop(self, image):
        """Crop the image using cached transform. Returns None if bbox is
        completely outside the image."""
        img_h, img_w = image.shape[:2]
        current_size = (img_h, img_w)
        current_hash = self._config_hash()

        # Recompute transform if size or config changed
        if (self._cached_output_size is None or
            self._cached_image_size != current_size or
            self._cached_config_hash != current_hash):
            self._compute_transform(img_h, img_w)

        if self._cached_matrix is None:
            return None

        return cv2.warpPerspective(image, self._cached_matrix, self._cached_output_size)

    @property
    def config(self):
        """Get current configuration."""
        return self._config.copy()

    @property
    def was_clamped(self):
        """True if the last transform required clamping."""
        return self._was_clamped

    @property
    def warnings(self):
        """Human-readable list of adjustments made during clamping."""
        return self._warnings.copy()
