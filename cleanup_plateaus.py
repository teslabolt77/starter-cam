import csv, os, shutil
from datetime import datetime

CSV_PATH = os.path.expanduser("~/starter_cam/starter_log.csv")

# Aggressive plateau removal tuning
UP_JUMP_PX = 80          # detect sudden jump up
DOWN_JUMP_PX = 80        # detect sudden jump down
MIN_PLATEAU_LEN = 3      # minimum consecutive points to consider a plateau
PLATEAU_NEAR_MAX_FRAC = 0.90  # plateau must sit near max to count as lid-lock

def backup_file(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_plateau_{ts}"
    shutil.copy2(path, bak)
    return bak

def to_int(x, default=0):
    try:
        return int(str(x).strip() or default)
    except Exception:
        return default

def main():
    if not os.path.exists(CSV_PATH):
        print("No CSV found:", CSV_PATH)
        return 1

    bak = backup_file(CSV_PATH)
    print("Backup created:", bak)

    with open(CSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames or []

    if "timestamp" not in fieldnames or "height_px" not in fieldnames:
        print("CSV missing required columns.")
        return 2

    if "rejected" not in fieldnames:
        fieldnames = fieldnames + ["rejected"]
        for row in rows:
            row["rejected"] = "0"

    # Build height list (None if bad)
    heights = []
    for row in rows:
        try:
            heights.append(float(row["height_px"]))
        except Exception:
            heights.append(None)

    valid = [h for h in heights if h is not None]
    if len(valid) < 10:
        print("Not enough data to clean.")
        return 0

    hmax = max(valid)
    near_max = PLATEAU_NEAR_MAX_FRAC * hmax

    # Helper: treat existing rejected as unavailable for baseline transitions
    rej = [to_int(row.get("rejected", "0")) == 1 for row in rows]

    marked = 0
    i = 1
    while i < len(heights):
        if heights[i] is None or heights[i-1] is None:
            i += 1
            continue

        # Look for a sharp jump up (accepted points only)
        if (not rej[i]) and (not rej[i-1]) and (heights[i] - heights[i-1] >= UP_JUMP_PX):
            start_idx = i

            # Plateau must be near max for lid-lock
            if heights[start_idx] < near_max:
                i += 1
                continue

            # Walk forward while values stay "high-ish" (still near max), allowing small noise
            j = start_idx
            while j + 1 < len(heights) and heights[j+1] is not None:
                if heights[j+1] >= near_max:
                    j += 1
                else:
                    break

            plateau_end = j

            # Need minimum length
            if plateau_end - start_idx + 1 < MIN_PLATEAU_LEN:
                i += 1
                continue

            # Now look for a sharp drop shortly after the plateau ends
            k = plateau_end + 1
            found_drop = False
            drop_idx = None
            # allow searching a few samples after plateau ends
            SEARCH_AFTER = 6
            while k < len(heights) and k <= plateau_end + SEARCH_AFTER:
                if heights[k] is None or heights[k-1] is None:
                    k += 1
                    continue
                if (not rej[k]) and (not rej[k-1]) and (heights[k-1] - heights[k] >= DOWN_JUMP_PX):
                    found_drop = True
                    drop_idx = k
                    break
                k += 1

            if found_drop:
                # Mark from start_idx through plateau_end as rejected (aggressive removal)
                for m in range(start_idx, plateau_end + 1):
                    if rows[m].get("rejected", "0") != "1":
                        rows[m]["rejected"] = "1"
                        marked += 1
                # Continue scanning after the drop point
                i = drop_idx + 1
                continue

        i += 1

    tmp_path = CSV_PATH + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp_path, CSV_PATH)
    print(f"Plateau cleanup done. Newly marked rejected: {marked}")
    print("Updated CSV:", CSV_PATH)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
