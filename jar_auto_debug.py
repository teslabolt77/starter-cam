#!/usr/bin/env python3
import os, glob, cv2, numpy as np

BASE = os.path.expanduser("~/starter_cam")
OUTDIR = BASE

def newest_run_image():
    # Prefer the non-overlay run frame (not edges/overlay)
    pats = [
        os.path.join(BASE, "photos_run", "run_*.jpg"),
        os.path.join(BASE, "photos_run", "*.jpg"),
        os.path.join(BASE, "raw_*.jpg"),
    ]
    cands = []
    for p in pats:
        cands += glob.glob(p)
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        raise SystemExit("No jpgs found (photos_run/run_*.jpg, photos_run/*.jpg, raw_*.jpg).")
    cands.sort(key=os.path.getmtime)
    return cands[-1]

def write(name, img):
    path = os.path.join(OUTDIR, name)
    cv2.imwrite(path, img)
    print("WROTE:", path)

img_path = newest_run_image()
img = cv2.imread(img_path)
if img is None:
    raise SystemExit(f"Failed to read {img_path}")

H, W = img.shape[:2]
print("Using:", img_path, f"({W}x{H})")

# --- Preprocess ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge debug (for your reference)
edges = cv2.Canny(gray_blur, 50, 150)
write("debug_edges.jpg", edges)

# --- “Dough-ish” mask via adaptive threshold (simple + fast) ---
# Invert so dough becomes white-ish region
th = cv2.adaptiveThreshold(
    gray_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51, 5
)

# Kill extreme margins (cardboard + right-side texture)
mx = int(W * 0.06)
th[:, :mx] = 0
th[:, W-mx:] = 0

# Kill top band (glass glare + residue) — keep dough zone
top_kill = int(H * 0.25)
th[:top_kill, :] = 0

write("debug_thresh_raw.png", th)

# --- KEY FIX: break thin vertical glass bridges BEFORE closing ---
# This is the “surgical” change you asked for.
vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
th = cv2.morphologyEx(th, cv2.MORPH_ERODE, vert_k, iterations=1)

# Now close to consolidate dough body
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

write("debug_thresh_fixed.png", th)

# Find contours of remaining blobs
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No contours found after filtering. (Try adjusting top_kill or kernel sizes.)")

# Pick "best dough" contour:
# - must be in lower half
# - prefer large area
# - prefer bottom proximity
best = None
best_score = -1e18

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area < 500:  # ignore specks
        continue

    cy = y + h/2
    if cy < H*0.45:  # too high = likely residue
        continue

    bottom = y + h
    bottom_score = bottom / H  # closer to bottom => larger
    area_score = np.log(area + 1.0)

    # penalize very tall skinny shapes (glass streaks)
    skinny_pen = 0.0
    if h > 0 and w > 0:
        ar = h / float(w)
        if ar > 3.0:
            skinny_pen = (ar - 3.0) * 1.2

    score = (3.0*bottom_score) + (2.0*area_score) - skinny_pen
    if score > best_score:
        best_score = score
        best = c

if best is None:
    raise SystemExit("Could not select a dough contour (all candidates rejected).")

# Draw contour debug
dbg_contour = img.copy()
cv2.drawContours(dbg_contour, [best], -1, (0,255,0), 2)
write("debug_dough_fill.png", dbg_contour)

# Estimate "surface y" as the highest point of the selected contour
# but only near the jar center (avoid side walls)
pts = best.reshape(-1,2)
x_center = W//2
band = int(W*0.25)
pts_mid = pts[(pts[:,0] >= x_center-band) & (pts[:,0] <= x_center+band)]
if pts_mid.shape[0] < 10:
    pts_mid = pts  # fallback

surface_y = int(np.min(pts_mid[:,1]))

# Overlay surface line
ov = img.copy()
cv2.line(ov, (0, surface_y), (W-1, surface_y), (0,255,0), 2)
cv2.putText(ov, f"surface_y={surface_y}", (10,25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
write("debug_surface_overlay.png", ov)

print("surface_y =", surface_y)
