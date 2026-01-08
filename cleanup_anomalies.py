import csv, os, shutil
from datetime import datetime

CSV_PATH = os.path.expanduser("~/starter_cam/starter_log.csv")

# TUNING (these work well for your lid-jump scenario)
MAX_JUMP_PX = 80          # reject if it jumps > 80 px from last accepted
HIGH_BAND_FRAC = 0.92     # "near the max" threshold for lid-lock candidates
NEIGHBOR_DROP_PX = 80     # if surrounding baseline is this much lower, mark as rejected
WINDOW = 3                # neighborhood size on each side for baseline check

def backup_file(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_{ts}"
    shutil.copy2(path, bak)
    return bak

def main():
    if not os.path.exists(CSV_PATH):
        print("No CSV found:", CSV_PATH)
        return 1

    bak = backup_file(CSV_PATH)
    print("Backup created:", bak)

    # Read all rows
    with open(CSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames or []

    if "timestamp" not in fieldnames or "height_px" not in fieldnames:
        print("CSV missing required columns.")
        return 2

    # Ensure rejected column exists
    if "rejected" not in fieldnames:
        fieldnames = fieldnames + ["rejected"]
        for row in rows:
            row["rejected"] = "0"

    # Convert heights
    heights = []
    for row in rows:
        try:
            heights.append(float(row["height_px"]))
        except Exception:
            heights.append(None)

    # Compute max valid
    valid = [h for h in heights if h is not None]
    if len(valid) < 5:
        print("Not enough data to clean.")
        return 0
    hmax = max(valid)
    high_band = HIGH_BAND_FRAC * hmax

    # Pass 1: big jump filter (stateful)
    rejected_jump = 0
    last_ok = None
    for i, h in enumerate(heights):
        if h is None:
            continue
        # if already rejected, don't use as last_ok baseline
        already_rej = int(str(rows[i].get("rejected","0")).strip() or "0") == 1
        if last_ok is None and not already_rej:
            last_ok = h
            continue

        if not already_rej and last_ok is not None and abs(h - last_ok) > MAX_JUMP_PX:
            rows[i]["rejected"] = "1"
            rejected_jump += 1
        else:
            # only update baseline if accepted
            if not already_rej and rows[i].get("rejected","0") != "1":
                last_ok = h

    # Pass 2: lid-lock plateau / high-band neighborhood filter
    # If a point is very high (near max) but neighbors are much lower, reject it.
    rejected_high = 0
    for i, h in enumerate(heights):
        if h is None:
            continue
        if rows[i].get("rejected","0") == "1":
            continue
        if h < high_band:
            continue

        neigh = []
        for j in range(max(0, i-WINDOW), min(len(heights), i+WINDOW+1)):
            if j == i:
                continue
            hj = heights[j]
            if hj is None:
                continue
            # prefer neighbors that are not rejected
            if rows[j].get("rejected","0") == "1":
                continue
            neigh.append(hj)

        if len(neigh) >= 2:
            baseline = sorted(neigh)[len(neigh)//2]  # median neighbor
            if (h - baseline) >= NEIGHBOR_DROP_PX:
                rows[i]["rejected"] = "1"
                rejected_high += 1

    # Write back
    tmp_path = CSV_PATH + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp_path, CSV_PATH)

    print(f"Cleanup done. Marked rejected via jump: {rejected_jump}, via high-band: {rejected_high}")
    print("Updated CSV:", CSV_PATH)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
