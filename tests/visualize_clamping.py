#!/usr/bin/env python3
"""
Visual test for bbox validation and clamping.

Displays a grid showing original vs clamped bounding boxes for various
edge-case configurations. Each cell shows the original bbox (red) and
the clamped result (green) side by side, with config details below.

Usage:
    python tests/visualize_clamping.py
    python tests/visualize_clamping.py --save output.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import compute_rotated_corners, clamp_config


# ── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = [
    ("Normal", {
        "alpha": 353.34, "ox": 0.434, "oy": 0.493,
        "width": 0.241, "height": 0.627,
    }),
    ("Negative w/h", {
        "alpha": 180, "ox": 0.8, "oy": 0.85,
        "width": -0.3, "height": -0.4,
    }),
    ("Corner OOB", {
        "alpha": 25, "ox": 0.15, "oy": 0.2,
        "width": 0.35, "height": 0.45,
    }),
    ("Oversized", {
        "alpha": 15, "ox": 0.5, "oy": 0.5,
        "width": 1.5, "height": 1.8,
    }),
    ("Outside image", {
        "alpha": 45, "ox": 2.5, "oy": -0.5,
        "width": 0.3, "height": 0.3,
    }),
    ("Huge angle", {
        "alpha": 7432.891, "ox": 0.123, "oy": 0.987,
        "width": 0.031, "height": 0.271,
    }),
]

# ── Layout constants ─────────────────────────────────────────────────────────
THUMB_W = 320          # each thumbnail width
THUMB_H = 240          # each thumbnail height
GAP = 20               # gap between cells
ARROW_W = 44           # width reserved for the arrow
LABEL_H = 32           # height for the cell label inside the cell
INFO_H = 90            # height for the text block under each pair
CELL_PAD = 8           # padding inside cell background
CELL_W = THUMB_W * 2 + ARROW_W + CELL_PAD * 2   # one cell (pair) width
CELL_H = LABEL_H + THUMB_H + INFO_H + CELL_PAD  # one cell height
COLS = 3               # grid columns
ROWS = 2               # grid rows
MARGIN = 28            # outer margin
TITLE_H = 60           # top title bar height

BG = (30, 30, 30)
BORDER_COLOR = (70, 70, 70)
ORIG_COLOR = (60, 80, 255)      # BGR red-orange
CLAMP_COLOR = (80, 220, 120)    # BGR green
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 160)
WARN_COLOR = (60, 200, 255)     # BGR yellow-orange


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_bbox_overlay(base_img, config, color):
    """Return a thumbnail with the rotated bbox drawn on the image."""
    img_h, img_w = base_img.shape[:2]
    thumb = cv2.resize(base_img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    # Dim the image slightly so the bbox stands out
    thumb = (thumb * 0.65).astype(np.uint8)

    sx = THUMB_W / img_w
    sy = THUMB_H / img_h

    alpha = config.get('alpha', 0)
    ox = config.get('ox', 0.5)
    oy = config.get('oy', 0.5)
    w = config.get('width', 0.5)
    h = config.get('height', 0.5)

    cx, cy = ox * img_w, oy * img_h
    pw, ph = abs(w) * img_w, abs(h) * img_h
    corners = compute_rotated_corners(cx, cy, pw, ph, alpha)

    sc = corners.copy()
    sc[:, 0] *= sx
    sc[:, 1] *= sy
    pts = sc.astype(np.int32)

    # Semi-transparent fill
    overlay = thumb.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.25, thumb, 0.75, 0, thumb)

    # Thick outline
    cv2.polylines(thumb, [pts], True, color, 2, cv2.LINE_AA)

    # Corner markers
    for pt in pts:
        cv2.circle(thumb, tuple(pt), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(thumb, tuple(pt), 4, color, 1, cv2.LINE_AA)

    # Center cross
    draw_cx = int(np.clip(ox * img_w * sx, 0, THUMB_W - 1))
    draw_cy = int(np.clip(oy * img_h * sy, 0, THUMB_H - 1))
    size = 8
    cv2.line(thumb, (draw_cx - size, draw_cy), (draw_cx + size, draw_cy),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(thumb, (draw_cx, draw_cy - size), (draw_cx, draw_cy + size),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(thumb, (draw_cx, draw_cy), 3, color, -1, cv2.LINE_AA)

    # Thin border around thumbnail
    cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), BORDER_COLOR, 1)

    return thumb


def put_text(canvas, text, org, scale=0.45, color=TEXT_WHITE, thickness=1):
    """Draw anti-aliased text."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_arrow(canvas, x, y_center):
    """Draw a right-pointing arrow."""
    y = y_center
    hw = ARROW_W // 2
    tip = 10
    cv2.line(canvas, (x + 4, y), (x + ARROW_W - 4, y), TEXT_DIM, 2, cv2.LINE_AA)
    cv2.line(canvas, (x + ARROW_W - 4 - tip, y - tip),
             (x + ARROW_W - 4, y), TEXT_DIM, 2, cv2.LINE_AA)
    cv2.line(canvas, (x + ARROW_W - 4 - tip, y + tip),
             (x + ARROW_W - 4, y), TEXT_DIM, 2, cv2.LINE_AA)


def fmt_config(cfg):
    """Format config values into short strings."""
    a = cfg.get('alpha', 0)
    ox = cfg.get('ox', 0.5)
    oy = cfg.get('oy', 0.5)
    w = cfg.get('width', 0.5)
    h = cfg.get('height', 0.5)
    return (
        f"angle={a:.1f}",
        f"center=({ox:.3f}, {oy:.3f})",
        f"size=({w:.3f}, {h:.3f})",
    )


# ── Main composition ────────────────────────────────────────────────────────

def build_visualization(base_img):
    """Build the full grid image."""
    img_h, img_w = base_img.shape[:2]

    total_w = MARGIN * 2 + COLS * CELL_W + (COLS - 1) * GAP
    total_h = MARGIN + TITLE_H + ROWS * CELL_H + (ROWS - 1) * GAP * 2 + MARGIN

    canvas = np.full((total_h, total_w, 3), BG[0], dtype=np.uint8)

    # ── Title bar ────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (total_w, TITLE_H), (45, 45, 45), -1)
    put_text(canvas, "BBOX CLAMPING VISUALIZATION", (MARGIN, 38),
             scale=0.85, color=TEXT_WHITE, thickness=2)

    # Legend on right side of title
    lx = total_w - 420
    cv2.rectangle(canvas, (lx, 16), (lx + 14, 30), ORIG_COLOR, -1)
    put_text(canvas, "Original", (lx + 20, 29), scale=0.45, color=ORIG_COLOR)
    cv2.rectangle(canvas, (lx + 120, 16), (lx + 134, 30), CLAMP_COLOR, -1)
    put_text(canvas, "Clamped", (lx + 140, 29), scale=0.45, color=CLAMP_COLOR)

    # ── Grid cells ───────────────────────────────────────────────────────
    for idx, (label, config) in enumerate(CONFIGS):
        row = idx // COLS
        col = idx % COLS

        x0 = MARGIN + col * (CELL_W + GAP)
        y0 = TITLE_H + MARGIN + row * (CELL_H + GAP)

        # Cell background with rounded-look border
        cv2.rectangle(canvas,
                      (x0, y0),
                      (x0 + CELL_W, y0 + CELL_H),
                      (48, 48, 48), -1)
        cv2.rectangle(canvas,
                      (x0, y0),
                      (x0 + CELL_W, y0 + CELL_H),
                      (65, 65, 65), 1)

        # ── Cell label (inside the cell, top strip) ──────────────────────
        cv2.rectangle(canvas,
                      (x0 + 1, y0 + 1),
                      (x0 + CELL_W - 1, y0 + LABEL_H),
                      (55, 55, 55), -1)
        put_text(canvas, f"{idx + 1}. {label}", (x0 + CELL_PAD, y0 + 22),
                 scale=0.55, color=TEXT_WHITE, thickness=1)

        # Thumbnail y-start
        ty = y0 + LABEL_H
        tx_orig = x0 + CELL_PAD
        tx_clamp = x0 + CELL_PAD + THUMB_W + ARROW_W

        # ── Original thumbnail ───────────────────────────────────────────
        orig_thumb = draw_bbox_overlay(base_img, config, ORIG_COLOR)
        canvas[ty:ty + THUMB_H, tx_orig:tx_orig + THUMB_W] = orig_thumb

        # ── Arrow ────────────────────────────────────────────────────────
        arrow_x = tx_orig + THUMB_W
        draw_arrow(canvas, arrow_x, ty + THUMB_H // 2)

        # ── Clamped thumbnail ────────────────────────────────────────────
        clamped_cfg, was_clamped, warnings = clamp_config(config, img_w, img_h)
        clamp_thumb = draw_bbox_overlay(base_img, clamped_cfg, CLAMP_COLOR)
        canvas[ty:ty + THUMB_H, tx_clamp:tx_clamp + THUMB_W] = clamp_thumb

        # ── Info block below thumbnails ──────────────────────────────────
        info_y = ty + THUMB_H + 4

        # Thin separator line
        cv2.line(canvas,
                 (x0 + CELL_PAD, info_y),
                 (x0 + CELL_W - CELL_PAD, info_y),
                 (65, 65, 65), 1)
        info_y += 4

        lines_orig = fmt_config(config)
        lines_clamp = fmt_config(clamped_cfg)

        # Original config info (left side, dimmed)
        for j, line in enumerate(lines_orig):
            put_text(canvas, line, (tx_orig + 2, info_y + 14 + j * 17),
                     scale=0.40, color=TEXT_DIM)

        # Clamped config info (right side, colored if changed)
        for j, line in enumerate(lines_clamp):
            clr = CLAMP_COLOR if was_clamped else TEXT_DIM
            put_text(canvas, line, (tx_clamp + 2, info_y + 14 + j * 17),
                     scale=0.40, color=clr)

        # Warnings (below clamped info, if any)
        if warnings:
            wy = info_y + 14 + len(lines_clamp) * 17 + 2
            for w_line in warnings[:2]:
                short = w_line[:50] + ("..." if len(w_line) > 50 else "")
                put_text(canvas, short, (tx_clamp + 2, wy),
                         scale=0.33, color=WARN_COLOR)
                wy += 14

        # Status badge (top-right of clamped thumbnail)
        if was_clamped:
            badge = "CLAMPED"
            badge_color = (0, 160, 240)
        else:
            badge = "OK"
            badge_color = (60, 180, 60)

        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx = tx_clamp + THUMB_W - tw - 12
        by = ty + 6
        cv2.rectangle(canvas, (bx - 5, by - 3), (bx + tw + 5, by + th + 5),
                      badge_color, -1)
        put_text(canvas, badge, (bx, by + th + 1), scale=0.42,
                 color=(0, 0, 0), thickness=1)

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bbox clamping for edge-case configurations"
    )
    parser.add_argument('--save', type=str, default=None,
                        help='Save output to file instead of displaying')
    parser.add_argument('--image', type=str, default=None,
                        help='Base image to use (default: img/pattern.jpg)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    if args.image:
        img_path = Path(args.image)
    else:
        img_path = project_root / "img" / "pattern.jpg"

    base_img = cv2.imread(str(img_path))
    if base_img is None:
        print(f"Warning: Cannot load {img_path}, using synthetic pattern")
        base_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for ci, (color, pos) in enumerate([
            ((255, 200, 0), (0, 0)), ((0, 0, 200), (320, 0)),
            ((0, 200, 0), (0, 240)), ((200, 0, 200), (320, 240)),
        ]):
            cv2.rectangle(base_img, pos, (pos[0] + 320, pos[1] + 240), color, -1)
        cv2.line(base_img, (0, 0), (640, 480), (255, 255, 255), 3)
        cv2.drawMarker(base_img, (320, 240), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)

    canvas = build_visualization(base_img)

    if args.save:
        cv2.imwrite(args.save, canvas)
        print(f"Saved to {args.save} ({canvas.shape[1]}x{canvas.shape[0]})")
    else:
        win = "Bbox Clamping Visualization"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(canvas.shape[1], 1920),
                         min(canvas.shape[0], 1080))
        cv2.imshow(win, canvas)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
