import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime

# ---- CONFIG ----
CAM_INDEX = 0
INTERVAL_S = 20

# ROI: x, y, w, h  (tune once)
ROI = (420, 120, 380, 520)

SAVE_FRAMES = True
FRAME_DIR = "frames"
CSV_PATH = "starter_log.csv"

# Ignore top/bottom of ROI where rim/base/glare live
SEARCH_TOP_FRAC = 0.15
SEARCH_BOT_FRAC = 0.90
# ----------------

def detect_surface_y(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Horizontal boundary strength (Sobel in Y)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_strength = np.mean(np.abs(sobel_y), axis=1)

    h = row_strength.shape[0]
    y0 = int(SEARCH_TOP_FRAC * h)
    y1 = int(SEARCH_BOT_FRAC * h)

    y = y0 + int(np.argmax(row_strength[y0:y1]))

    y_peak = y0 + int(np.argmax(row_strength[y0:y1]))
    y_stable = min(y_peak + int(0.12 * h), y1 - 1)  # shift down ~12% of ROI height
    print(f"y_peak={y_peak}, y_stable={y_stable}, roi_h={h}")
    return y_stable


def main():
    if SAVE_FRAMES:
        os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    # C920X likes 1080p; you can drop to 720p for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up camera exposure/white balance
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "height_px"])

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(2)
                continue

            x, y, w, h = ROI
            roi = frame[y:y+h, x:x+w]

            surface_y = detect_surface_y(roi)
            height_px = h - surface_y  # height from bottom of ROI

            ts = datetime.now().isoformat(timespec="seconds")
            writer.writerow([ts, height_px])
            f.flush()

            if SAVE_FRAMES:
                # Save a debug frame occasionally (or every time if you want)
                out = frame.copy()
                cv2.rectangle(out, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.line(out, (x, y + surface_y), (x+w, y + surface_y), (255, 255, 255), 2)
                cv2.imwrite(os.path.join(FRAME_DIR, f"{ts.replace(':','-')}.jpg"), out)

            print(ts, "height_px =", height_px)
            time.sleep(INTERVAL_S)

if __name__ == "__main__":
    main()
