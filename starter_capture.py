#!/usr/bin/env python3
"""
starter_capture.py — Sourdough starter surface detector (stable exposure + correct overlay)

Overlay behavior (FIXED):
- Yellow lines = JAR bounds (top + bottom) for visualization
- Green line  = detected dough surface
Detection behavior:
- Still uses a separate internal ROI band (search_top/search_bot) to avoid lid/rim
"""

import argparse, os, time, shutil, subprocess
from typing import Optional, Tuple

import cv2
import numpy as np

# -------- Defaults (tune these to match your red lines) --------
# JAR bounds are *visual only* (yellow lines)
DEFAULT_JAR_TOP_FRAC = 0.20
DEFAULT_JAR_BOT_FRAC = 0.92

# Detection ROI band (internal; keep below lid/rim)
DEFAULT_SEARCH_TOP_FRAC = 0.40
DEFAULT_SEARCH_BOT_FRAC = 0.80

ROW_SMOOTH_K = 21
GRAY_BLUR = 9
DEFAULT_MIN_CONF = 1.0
DEFAULT_DEBUG = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def draw_timestamp(img: np.ndarray, text: str) -> None:
    """Draw bottom-right timestamp with outline."""
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    h, w = img.shape[:2]
    x = w - tw - pad
    y = h - pad

    # black outline
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    # white text
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def have_v4l2ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None

def run_v4l2ctl(args_list) -> None:
    try:
        subprocess.run(["v4l2-ctl", *args_list], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def apply_camera_controls(device_index: int) -> None:
    """
    Uses the values you locked in via systemd, but harmless if called again.
    """
    if not have_v4l2ctl():
        return
    dev = f"/dev/video{device_index}"
    run_v4l2ctl(["-d", dev, "-c", "auto_exposure=1"])
    run_v4l2ctl(["-d", dev, "-c", "exposure_time_absolute=2000"])
    run_v4l2ctl(["-d", dev, "-c", "gain=200"])
    run_v4l2ctl(["-d", dev, "-c", "focus_automatic_continuous=0"])
    run_v4l2ctl(["-d", dev, "-c", "white_balance_automatic=1"])

def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def pick_surface_row_intensity(roi_bgr: np.ndarray) -> Tuple[int, float]:
    """
    Find surface by strongest downward intensity change.
    Returns (row_idx_rel, conf_norm).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    k = _odd(GRAY_BLUR)
    if k >= 3:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    row_means = np.mean(gray, axis=1)
    diff = row_means[:-1] - row_means[1:]

    sk = _odd(ROW_SMOOTH_K)
    if sk >= 3 and diff.shape[0] >= sk:
        diff = cv2.GaussianBlur(diff.reshape(-1, 1), (1, sk), 0).ravel()

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


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)

    ap.add_argument("--save-image", action="store_true")
    ap.add_argument("--image-dir", default="/home/pi/starter_cam/captures")
    ap.add_argument("--prefix", default="starter")

    # Detection ROI band (internal)
    ap.add_argument("--search-top-frac", type=float, default=DEFAULT_SEARCH_TOP_FRAC)
    ap.add_argument("--search-bot-frac", type=float, default=DEFAULT_SEARCH_BOT_FRAC)

    # Jar bounds overlay (yellow lines)
    ap.add_argument("--jar-top-frac", type=float, default=DEFAULT_JAR_TOP_FRAC)
    ap.add_argument("--jar-bot-frac", type=float, default=DEFAULT_JAR_BOT_FRAC)

    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)

    ap.add_argument("--debug", action="store_true", default=DEFAULT_DEBUG)
    ap.add_argument("--no-debug", action="store_true")

    args = ap.parse_args()
    debug = args.debug and (not args.no_debug)

    ensure_dir(args.image_dir)

    # (Optional) re-apply known-good camera settings (you also have systemd doing this)
    apply_camera_controls(args.device)

    cap = open_camera(args.device, args.width, args.height)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return 1

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("ERROR: Could not read frame.")
        return 2

    H, W = frame.shape[:2]

    # Detection ROI (internal)
    y1 = int(np.clip(H * args.search_top_frac, 0, H - 2))
    y2 = int(np.clip(H * args.search_bot_frac, y1 + 1, H - 1))
    roi = frame[y1:y2, :].copy()

    row_rel, conf = pick_surface_row_intensity(roi)
    rejected = conf < args.min_conf
    if rejected:
        row_rel = roi.shape[0] // 2
    surface_y = y1 + row_rel

    # Jar overlay bounds (visual only)
    jar_top = int(np.clip(H * args.jar_top_frac, 0, H - 2))
    jar_bot = int(np.clip(H * args.jar_bot_frac, jar_top + 1, H - 1))

    if args.save_image:
        img_path = os.path.join(args.image_dir, f"{args.prefix}_{now_stamp()}.jpg")
        cv2.imwrite(img_path, frame)
        print(img_path)

    if debug:
        roi_path = os.path.join(args.image_dir, f"{args.prefix}_roi_{now_stamp()}.jpg")
        cv2.imwrite(roi_path, roi)

        overlay = frame.copy()
        ts_label = time.strftime("%Y-%m-%d %H:%M:%S")
        draw_timestamp(overlay, ts_label)

        # Yellow = jar bounds (what you drew in red)
        cv2.line(overlay, (0, jar_top), (W - 1, jar_top), (0, 255, 255), 1)
        cv2.line(overlay, (0, jar_bot), (W - 1, jar_bot), (0, 255, 255), 1)

        # Green = detected surface
        cv2.line(overlay, (0, surface_y), (W - 1, surface_y), (0, 255, 0), 2)

        label = f"surface_y={surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]"
        if rejected:
            label += " REJECTED"
        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ov_path = os.path.join(args.image_dir, f"{args.prefix}_overlay_{now_stamp()}.jpg")
        cv2.imwrite(ov_path, overlay)

    print(f"SURFACE_Y {surface_y} conf={conf:.2f} ROI[{y1}:{y2}] JAR[{jar_top}:{jar_bot}]{' REJECTED' if rejected else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
