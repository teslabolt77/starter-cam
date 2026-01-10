#!/usr/bin/env python3
import os, time, argparse
from typing import Tuple
import numpy as np
import cv2

# ---------------- Defaults ----------------
DEFAULT_DEVICE = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Manual jar fractions (fallback if auto-jar fails)
DEFAULT_JAR_TOP_FRAC = 0.20
DEFAULT_JAR_BOT_FRAC = 0.99

# Search ROI inside jar (fractions of jar height)
DEFAULT_SEARCH_TOP_FRAC = 0.62
DEFAULT_SEARCH_BOT_FRAC = 0.97

DEFAULT_ROI_PAD_FRAC = 0.12
DEFAULT_MIN_CONF = 0.60

# "Dough never starts below this fraction up from jar bottom"
# (0.20 = surface must be at least 20% above the jar bottom)
DEFAULT_MIN_FROM_BOTTOM_FRAC = 0.20

# Smoothing
GRAY_BLUR = 5
ROW_SMOOTH_K = 31

# Auto-jar tuning
AUTOJAR_CENTER_X_FRAC = 0.15  # ignore outer margins when finding jar
AUTOJAR_MIN_W_FRAC = 0.25
AUTOJAR_MAX_W_FRAC = 0.85
AUTOJAR_MIN_H_FRAC = 0.40
AUTOJAR_MAX_H_FRAC = 0.99

# ------------------------------------------


def _odd(k: int) -> int:
    k = int(k)
    if k < 0:
        return 0
    return k if (k % 2 == 1) else (k + 1)


def pick_surface_row_intensity(roi_bgr: np.ndarray, pad_rows: int = 0) -> Tuple[int, float]:
    """
    Find surface by strongest downward intensity change (row-to-row).
    Returns (row_idx_rel, conf_norm).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    k = _odd(GRAY_BLUR)
    if k >= 3:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    row_means = np.mean(gray, axis=1)
    diff = row_means[:-1] - row_means[1:]  # positive = bright->dark transition

    sk = _odd(ROW_SMOOTH_K)
    if sk >= 3 and diff.shape[0] >= sk:
        diff = cv2.GaussianBlur(diff.reshape(-1, 1), (1, sk), 0).ravel()

    # Ignore ROI edges (rim/shoulder/bottom edges often dominate)
    if pad_rows > 0 and diff.shape[0] > (2 * pad_rows + 2):
        diff[:pad_rows] = 0
        diff[-pad_rows:] = 0

    peak_idx = int(np.argmax(diff))
    peak_val = float(diff[peak_idx])

    baseline = float(np.median(diff))
    mad = float(np.median(np.abs(diff - baseline))) + 1e-6
    conf_norm = (peak_val - baseline) / (1.4826 * mad)
    return peak_idx, conf_norm


def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def auto_detect_jar_bbox(frame_bgr: np.ndarray):
    """
    Return (x1,x2,y1,y2, score) for jar bbox in image coordinates, or None if not found.
    Strategy:
      - edge map -> dilate -> contours
      - choose a big, centered contour with plausible jar dimensions
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)

    # Ignore outer margins (cardboard edges & right-side clutter)
    xm = int(W * AUTOJAR_CENTER_X_FRAC)
    edges[:, :xm] = 0
    edges[:, W - xm:] = 0

    # Close gaps so jar outline becomes a more contiguous contour
    kernel = np.ones((5, 5), np.uint8)
    edges2 = cv2.dilate(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, edges, edges2

    cx = W / 2.0

    best = None
    best_score = -1e18

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        # size filters
        if w < W * AUTOJAR_MIN_W_FRAC or w > W * AUTOJAR_MAX_W_FRAC:
            continue
        if h < H * AUTOJAR_MIN_H_FRAC or h > H * AUTOJAR_MAX_H_FRAC:
            continue

        # jar is tall-ish
        aspect = h / float(w)
        if aspect < 0.7 or aspect > 3.5:
            continue

        # centered preference
        jar_cx = x + w / 2.0
        center_penalty = abs(jar_cx - cx) / W  # 0 is perfect

        area = w * h
        score = area - (center_penalty * area * 0.35)

        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)

    if best is None:
        return None, edges, edges2

    x1, y1, x2, y2 = best

    # Expand a bit to include rim thickness / faint sides
    pad_x = int(0.04 * (x2 - x1))
    pad_y = int(0.02 * (y2 - y1))

    x1 = max(0, x1 - pad_x)
    x2 = min(W - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(H - 1, y2 + pad_y)

    return (x1, x2, y1, y2, float(best_score)), edges, edges2


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)

    ap.add_argument("--save-image", action="store_true")
    ap.add_argument("--image-dir", type=str, default=".")
    ap.add_argument("--prefix", type=str, default="run")

    # Search ROI inside jar
    ap.add_argument("--search-top-frac", type=float, default=DEFAULT_SEARCH_TOP_FRAC)
    ap.add_argument("--search-bot-frac", type=float, default=DEFAULT_SEARCH_BOT_FRAC)
    ap.add_argument("--roi-pad-frac", type=float, default=DEFAULT_ROI_PAD_FRAC)
    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)

    # Manual jar fallback
    ap.add_argument("--jar-top-frac", type=float, default=DEFAULT_JAR_TOP_FRAC)
    ap.add_argument("--jar-bot-frac", type=float, default=DEFAULT_JAR_BOT_FRAC)

    # Hard rule: surface must be at least X up from bottom
    ap.add_argument("--min-from-bottom-frac", type=float, default=DEFAULT_MIN_FROM_BOTTOM_FRAC)

    # Debug images
    ap.add_argument("--debug", action="store_true", default=True)
    ap.add_argument("--no-debug", action="store_false", dest="debug")

    args = ap.parse_args()
    debug = bool(args.debug)

    os.makedirs(args.image_dir, exist_ok=True)

    cap = open_camera(args.device, args.width, args.height)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("ERROR: Could not read frame.")
        return 2

    H, W = frame.shape[:2]

    # ---------- AUTOJAR: detect jar bbox ----------
    auto = auto_detect_jar_bbox(frame)
    used_autojar = False

    if auto and auto[0] is not None:
        (jx1, jx2, jy1, jy2, _score), edges, edges2 = auto
        jar_top = int(np.clip(jy1, 0, H - 2))
        jar_bot = int(np.clip(jy2, jar_top + 1, H - 1))
        jar_x1 = int(np.clip(jx1, 0, W - 2))
        jar_x2 = int(np.clip(jx2, jar_x1 + 1, W - 1))
        used_autojar = True
    else:
        edges = None
        edges2 = None
        jar_top = int(np.clip(H * args.jar_top_frac, 0, H - 2))
        jar_bot = int(np.clip(H * args.jar_bot_frac, jar_top + 1, H - 1))
        jar_x1, jar_x2 = 0, W - 1

    jar_h = max(1, (jar_bot - jar_top))

    # Detection ROI (internal) - relative to jar bounds
    y1 = int(np.clip(jar_top + jar_h * args.search_top_frac, jar_top, jar_bot - 2))
    y2 = int(np.clip(jar_top + jar_h * args.search_bot_frac, y1 + 1, jar_bot - 1))

    # ROI inside jar X bounds (reduces cardboard/right-side clutter)
    roi = frame[y1:y2, jar_x1:jar_x2].copy()

    # Detect surface within ROI, ignoring ROI edge rows
    pad_rows = int(max(0, min(roi.shape[0] // 3, roi.shape[0] * args.roi_pad_frac)))
    row_rel, conf = pick_surface_row_intensity(roi, pad_rows=pad_rows)

    surface_y = y1 + row_rel

    # Enforce "must be at least X up from bottom of jar"
    min_y_allowed = int(jar_bot - jar_h * float(args.min_from_bottom_frac))
    clamped = False
    if surface_y > min_y_allowed:
        surface_y = min_y_allowed
        clamped = True

    rejected = (conf < args.min_conf)

    # if rejected, still show something stable: use midpoint of ROI but keep clamp rule
    if rejected:
        surface_y = y1 + (roi.shape[0] // 2)
        if surface_y > min_y_allowed:
            surface_y = min_y_allowed
            clamped = True

    # ---------- save raw ----------
    if args.save_image:
        img_path = os.path.join(args.image_dir, f"{args.prefix}_{now_stamp()}.jpg")
        cv2.imwrite(img_path, frame)
        print(img_path)

    # ---------- debug images ----------
    if debug and used_autojar and edges is not None:
        dbg_edges_path = os.path.join(args.image_dir, f"{args.prefix}_debug_edges_{now_stamp()}.jpg")
        cv2.imwrite(dbg_edges_path, edges)

        dbg2 = frame.copy()
        # jar bbox in yellow
        cv2.rectangle(dbg2, (jar_x1, jar_top), (jar_x2, jar_bot), (0, 255, 255), 1)
        # ROI in yellow
        cv2.rectangle(dbg2, (jar_x1, y1), (jar_x2, y2), (0, 255, 255), 1)
        # surface in green
        cv2.line(dbg2, (0, surface_y), (W - 1, surface_y), (0, 255, 0), 2)
        dbg_autojar_path = os.path.join(args.image_dir, f"{args.prefix}_debug_autojar_{now_stamp()}.jpg")
        cv2.imwrite(dbg_autojar_path, dbg2)

    if debug:
        # ROI image
        roi_path = os.path.join(args.image_dir, f"{args.prefix}_roi_{now_stamp()}.jpg")
        cv2.imwrite(roi_path, roi)

        # overlay
        overlay = frame.copy()

        # jar bounds (yellow lines)
        cv2.line(overlay, (0, jar_top), (W - 1, jar_top), (0, 255, 255), 1)
        cv2.line(overlay, (0, jar_bot), (W - 1, jar_bot), (0, 255, 255), 1)

        # surface line (green)
        cv2.line(overlay, (0, surface_y), (W - 1, surface_y), (0, 255, 0), 2)

        ts_label = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay, ts_label, (W - 260, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        extra = " AUTOJAR" if used_autojar else ""
        if clamped:
            extra += " CLAMP"
        label = f"surface_y={surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]{extra}"
        cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        overlay_path = os.path.join(args.image_dir, f"{args.prefix}_overlay_{now_stamp()}.jpg")
        cv2.imwrite(overlay_path, overlay)

    # stdout parse line for runner/app
    extra2 = " AUTOJAR" if used_autojar else ""
    if clamped:
        extra2 += " CLAMP"
    print(f"SURFACE_Y {surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]{' REJECTED' if rejected else ''}{extra2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
import argparse
import os
import time
from typing import Tuple, Optional

import cv2
import numpy as np

# -------------------------
# Defaults / tunables
# -------------------------

DEFAULT_DEVICE = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Old "fixed" jar fractions (still supported as fallback)
DEFAULT_JAR_TOP_FRAC = 0.20
DEFAULT_JAR_BOT_FRAC = 0.99

# Detection ROI within jar (still supported)
DEFAULT_SEARCH_TOP_FRAC = 0.65
DEFAULT_SEARCH_BOT_FRAC = 0.92

# How much to ignore at top/bottom of ROI (old approach)
DEFAULT_ROI_PAD_FRAC = 0.08

DEFAULT_MIN_CONF = 0.60

# ---- Dough-mask detector knobs ----
# We want dough = "brightest low region" inside jar.
# These are robust across modest lighting changes.
DOUGH_BLUR_K = 7              # blur to reduce noise
DOUGH_TOPCUT_FRAC = 0.10      # ignore very top of jar (flour smears)
DOUGH_BOTTOMCUT_FRAC = 0.02   # ignore very bottom row(s)
DOUGH_MIN_AREA_FRAC = 0.010   # minimum blob area relative to jar area
DOUGH_MIN_WIDTH_FRAC = 0.18   # blob must be at least this wide relative to jar width
DOUGH_TOUCH_BOTTOM_FRAC = 0.22  # require blob touches within bottom X% of jar height

# AUTOJAR behavior
DEFAULT_AUTOJAR = 0  # 0/1
AUTOJAR_X_MARGIN_FRAC = 0.10
AUTOJAR_MIN_JAR_WIDTH_FRAC = 0.35  # sanity: jar must occupy at least this fraction of frame width

# Debug images
DEFAULT_DEBUG = True

# -------------------------
# Helpers
# -------------------------

def _odd(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    if k % 2 == 0:
        return k + 1
    return k

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def largest_component(mask: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Return (mask_of_largest_component, area)"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, 0
    # stats: [label, x,y,w,h,area], label 0 = background
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    comp = (labels == idx).astype(np.uint8) * 255
    return comp, area

def autojar_bounds(frame_bgr: np.ndarray, debug: bool = False, dbg_prefix: str = "debug") -> Tuple[int,int,int,int,float]:
    """
    Try to find jar bounds automatically.
    Returns: (x1, x2, jar_top, jar_bot, conf)
    If fails, returns full-frame X and default fractions for Y (low conf).
    """
    H, W = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (_odd(5), _odd(5)), 0)

    # Edge map
    edges = cv2.Canny(gray, 40, 140)

    # Ignore outer margins so we don't pick the right cardboard/mesh edge
    xm = int(W * AUTOJAR_X_MARGIN_FRAC)
    edges[:, :xm] = 0
    edges[:, W-xm:] = 0

    # Sum vertical edge energy per column
    col_energy = edges.sum(axis=0).astype(np.float32)
    col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (41, 1), 0).ravel()

    # Heuristic: jar is centered-ish. Find two strong peaks around center.
    cx = W // 2
    left_band = col_energy[:cx]
    right_band = col_energy[cx:]

    # If nothing meaningful, fail
    if left_band.max() <= 0 or right_band.max() <= 0:
        return 0, W-1, int(H * DEFAULT_JAR_TOP_FRAC), int(H * DEFAULT_JAR_BOT_FRAC), 0.0

    lx = int(np.argmax(left_band))
    rx = int(cx + np.argmax(right_band))

    # Sanity: jar width must be reasonable
    jar_w = rx - lx
    if jar_w < int(W * AUTOJAR_MIN_JAR_WIDTH_FRAC) or jar_w < 80:
        return 0, W-1, int(H * DEFAULT_JAR_TOP_FRAC), int(H * DEFAULT_JAR_BOT_FRAC), 0.0

    # Now detect jar top/bottom using horizontal edge energy inside that x-range.
    x1 = max(0, min(lx, W-2))
    x2 = max(x1+1, min(rx, W-1))

    roi = edges[:, x1:x2]
    row_energy = roi.sum(axis=1).astype(np.float32)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 41), 0).ravel()

    mid = H // 2
    top_band = row_energy[:mid]
    bot_band = row_energy[mid:]

    jar_top = int(np.argmax(top_band))
    jar_bot = int(mid + np.argmax(bot_band))

    # Clamp and sanity
    jar_top = max(0, min(jar_top, H-2))
    jar_bot = max(jar_top+1, min(jar_bot, H-1))

    # Confidence based on width + edge energy separation
    conf = float(jar_w / max(1, W))
    conf *= 1.0 if jar_bot - jar_top > int(H * 0.45) else 0.6  # jar should be tall

    if debug:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.line(dbg, (x1, 0), (x1, H-1), (0,255,255), 1)
        cv2.line(dbg, (x2, 0), (x2, H-1), (0,255,255), 1)
        cv2.line(dbg, (0, jar_top), (W-1, jar_top), (0,255,255), 1)
        cv2.line(dbg, (0, jar_bot), (W-1, jar_bot), (0,255,255), 1)
        cv2.imwrite(os.path.join(args_image_dir_global(), f"{dbg_prefix}_debug_autojar_{now_stamp()}.jpg"), dbg)

    return x1, x2, jar_top, jar_bot, float(np.clip(conf, 0.0, 1.0))

# We use a tiny trick so autojar debug can write into --image-dir without global state pollution
_IMAGE_DIR_FOR_DEBUG = None
def args_image_dir_global():
    return _IMAGE_DIR_FOR_DEBUG or "."

def detect_surface_by_doughmask(frame_bgr: np.ndarray,
                                jar_top: int, jar_bot: int,
                                x1: int, x2: int,
                                debug: bool = False,
                                prefix: str = "run") -> Tuple[int, float, dict]:
    """
    Detect the dough surface by segmenting the bright dough region inside jar bounds,
    and taking the top-most row of the chosen blob.
    Returns: (surface_y, conf, extras)
    """
    H, W = frame_bgr.shape[:2]
    jar_top = int(np.clip(jar_top, 0, H-2))
    jar_bot = int(np.clip(jar_bot, jar_top+1, H-1))
    x1 = int(np.clip(x1, 0, W-2))
    x2 = int(np.clip(x2, x1+1, W-1))

    jar = frame_bgr[jar_top:jar_bot, x1:x2].copy()
    jH, jW = jar.shape[:2]

    # Ignore tiny top strip where flour smears live
    yA = int(np.clip(jH * DOUGH_TOPCUT_FRAC, 0, jH-2))
    yB = int(np.clip(jH * (1.0 - DOUGH_BOTTOMCUT_FRAC), yA+1, jH))
    jar_work = jar[yA:yB, :]

    gray = cv2.cvtColor(jar_work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (_odd(DOUGH_BLUR_K), _odd(DOUGH_BLUR_K)), 0)

    # Otsu threshold on "bright = dough"
    # If lighting is dark, Otsu will still separate bright dough from darker glass/background.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up: close holes, then open small specks
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)

    # Keep only pixels that are in the lower-ish region (dough lives low)
    # This prevents flour patches high on jar from being selected.
    low_gate = np.zeros_like(th)
    low_start = int(th.shape[0] * (1.0 - DOUGH_TOUCH_BOTTOM_FRAC))
    low_gate[low_start:, :] = 255
    th_low = cv2.bitwise_and(th, low_gate)

    # Pick the largest connected component in th_low, but apply shape sanity checks
    comp, area = largest_component(th_low)
    extras = {
        "blob_area": area,
        "jar_w": jW,
        "jar_h": jH,
        "x1": x1, "x2": x2,
        "jar_top": jar_top, "jar_bot": jar_bot,
        "yA": yA, "yB": yB
    }

    if comp is None or area <= 0:
        # Fallback: center of search region, low confidence
        return int(jar_top + (jar_bot-jar_top) * 0.70), 0.0, extras

    # Bounding box + width sanity
    ys, xs = np.where(comp > 0)
    if len(ys) == 0:
        return int(jar_top + (jar_bot-jar_top) * 0.70), 0.0, extras

    bx1, bx2 = int(xs.min()), int(xs.max())
    by1, by2 = int(ys.min()), int(ys.max())
    bw = max(1, bx2 - bx1 + 1)
    bh = max(1, by2 - by1 + 1)

    jar_area = float(th.shape[0] * th.shape[1])
    area_frac = float(area) / max(1.0, jar_area)
    width_frac = float(bw) / max(1.0, float(th.shape[1]))

    # Surface = top row of blob in jar_work coords
    surface_rel_in_work = int(by1)
    surface_y = jar_top + yA + surface_rel_in_work

    # Confidence: weighted blend of area + width + vertical thickness (more stable = higher)
    conf = 0.0
    conf += np.clip((area_frac - DOUGH_MIN_AREA_FRAC) / (0.20), 0.0, 1.0) * 0.50
    conf += np.clip((width_frac - DOUGH_MIN_WIDTH_FRAC) / (0.40), 0.0, 1.0) * 0.35
    conf += np.clip(float(bh) / max(1.0, th.shape[0] * 0.35), 0.0, 1.0) * 0.15
    conf = float(np.clip(conf, 0.0, 1.0))

    # If it doesn't meet minimums, treat as rejected-ish
    if area_frac < DOUGH_MIN_AREA_FRAC or width_frac < DOUGH_MIN_WIDTH_FRAC:
        conf *= 0.35

    if debug:
        dbg = frame_bgr.copy()

        # Draw jar box
        cv2.rectangle(dbg, (x1, jar_top), (x2, jar_bot), (0,255,255), 1)

        # Draw dough bbox in frame coords
        rx1 = x1 + bx1
        rx2 = x1 + bx2
        ry1 = jar_top + yA + by1
        ry2 = jar_top + yA + by2
        cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (0,255,0), 2)

        # Draw surface line
        cv2.line(dbg, (0, surface_y), (W-1, surface_y), (0,255,0), 2)

        # Save debug artifacts
        cv2.imwrite(os.path.join(args_image_dir_global(), f"{prefix}_debug_autojar_{now_stamp()}.jpg"), dbg)

        # Also dump ROI + mask for inspection
        cv2.imwrite(os.path.join(args_image_dir_global(), f"{prefix}_roi_{now_stamp()}.jpg"), jar_work)
        cv2.imwrite(os.path.join(args_image_dir_global(), f"{prefix}_mask_{now_stamp()}.jpg"), th_low)

    return surface_y, conf, extras

# -------------------------
# Main
# -------------------------

def main() -> int:
    global _IMAGE_DIR_FOR_DEBUG

    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)

    ap.add_argument("--save-image", action="store_true")
    ap.add_argument("--image-dir", type=str, default=".")
    ap.add_argument("--prefix", type=str, default="run")

    ap.add_argument("--search-top-frac", type=float, default=DEFAULT_SEARCH_TOP_FRAC)
    ap.add_argument("--search-bot-frac", type=float, default=DEFAULT_SEARCH_BOT_FRAC)
    ap.add_argument("--jar-top-frac", type=float, default=DEFAULT_JAR_TOP_FRAC)
    ap.add_argument("--jar-bot-frac", type=float, default=DEFAULT_JAR_BOT_FRAC)
    ap.add_argument("--roi-pad-frac", type=float, default=DEFAULT_ROI_PAD_FRAC)

    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)

    # New: enable autojar (find jar bounds based on edges)
    ap.add_argument("--autojar", type=int, default=DEFAULT_AUTOJAR, help="1=auto-detect jar bounds, 0=use jar-top/bot fracs")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-debug", action="store_true")

    args = ap.parse_args()

    debug = DEFAULT_DEBUG
    if args.debug:
        debug = True
    if args.no_debug:
        debug = False

    os.makedirs(args.image_dir, exist_ok=True)
    _IMAGE_DIR_FOR_DEBUG = args.image_dir

    cap = open_camera(args.device, args.width, args.height)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("ERROR: Could not read frame.")
        return 2

    H, W = frame.shape[:2]

    # Jar bounds
    if args.autojar == 1:
        x1, x2, jar_top, jar_bot, aj_conf = autojar_bounds(frame, debug=False)
        autojar_used = aj_conf > 0.0
    else:
        x1, x2 = 0, W-1
        jar_top = int(np.clip(H * args.jar_top_frac, 0, H - 2))
        jar_bot = int(np.clip(H * args.jar_bot_frac, jar_top + 1, H - 1))
        autojar_used = False

    # Detection ROI (internal) - relative to jar bounds (kept for printed ROI only)
    jar_h = max(1, (jar_bot - jar_top))
    y1 = int(np.clip(jar_top + jar_h * args.search_top_frac, jar_top, jar_bot - 2))
    y2 = int(np.clip(jar_top + jar_h * args.search_bot_frac, y1 + 1, jar_bot - 1))

    # ---- NEW surface detection ----
    surface_y, conf01, extras = detect_surface_by_doughmask(
        frame, jar_top, jar_bot, x1, x2,
        debug=debug,
        prefix=args.prefix
    )

    # Map conf01 to something more like your old "Z-score" scale (but stable)
    # (This keeps your dashboard behavior sane while you transition.)
    conf = float(np.clip(conf01 * 6.0, 0.0, 12.0))

    rejected = conf01 < args.min_conf

    # If rejected, choose a reasonable fallback: mid of ROI band, not bottom.
    if rejected:
        surface_y = int((y1 + y2) // 2)

    # Save raw image
    if args.save_image:
        img_path = os.path.join(args.image_dir, f"{args.prefix}_{now_stamp()}.jpg")
        cv2.imwrite(img_path, frame)
        print(img_path)

    # Overlay
    overlay = frame.copy()

    # Jar lines (yellow)
    cv2.line(overlay, (0, jar_top), (W-1, jar_top), (0, 255, 255), 1)
    cv2.line(overlay, (0, jar_bot), (W-1, jar_bot), (0, 255, 255), 1)

    # ROI band lines (yellow)
    cv2.line(overlay, (0, y1), (W-1, y1), (0, 255, 255), 1)
    cv2.line(overlay, (0, y2), (W-1, y2), (0, 255, 255), 1)

    # Surface line (green)
    cv2.line(overlay, (0, surface_y), (W-1, surface_y), (0, 255, 0), 2)

    label = f"surface_y={surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]"
    if autojar_used:
        label += " AUTOJAR"
    if rejected:
        label += " REJECTED"

    cv2.putText(overlay, label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    ts_label = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(overlay, ts_label, (W - 260, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    overlay_path = os.path.join(args.image_dir, f"{args.prefix}_overlay_{now_stamp()}.jpg")
    cv2.imwrite(overlay_path, overlay)

    # Print parse line (used by capture_runner.py)
    print(
        f"SURFACE_Y {surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]"
        f"{' AUTOJAR' if autojar_used else ''}"
        f"{' REJECTED' if rejected else ''}"
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
