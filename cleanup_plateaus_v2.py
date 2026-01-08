import csv, os, shutil
from datetime import datetime

CSV_PATH = os.path.expanduser("~/starter_cam/starter_log.csv")

# Tunables (aggressive)
UP_JUMP_PX = 70          # detect jump up
DOWN_JUMP_PX = 70        # detect drop down
LOOKAHEAD = 90           # how many points ahead to search for the drop (90 points ~ 7.5h if 5-min photos; ~45h if 30-min)
ABOVE_BASE_PX = 80       # points must be this far above baseline to be considered "in the lid-lock zone"
MIN_LEN = 3              # minimum length to reject

def backup_file(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_plateau2_{ts}"
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

    heights = []
    for row in rows:
        try:
            heights.append(float(row["height_px"]))
        except Exception:
            heights.append(None)

    rej = [to_int(row.get("rejected","0")) == 1 for row in rows]

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n//2] if n else None

    marked = 0
    i = 1
    while i < len(heights):
        if heights[i] is None or heights[i-1] is None:
            i += 1
            continue
        if rej[i] or rej[i-1]:
            i += 1
            continue

        # Detect jump up
        if heights[i] - heights[i-1] >= UP_JUMP_PX:
            up_idx = i

            # Baseline = median of a few points before the jump (accepted only)
            prev = []
            for j in range(max(0, up_idx-6), up_idx):
                if heights[j] is not None and not rej[j]:
                    prev.append(heights[j])
            base = median(prev)
            if base is None:
                i += 1
                continue

            # Find a later drop down within LOOKAHEAD
            drop_idx = None
            for k in range(up_idx+1, min(len(heights), up_idx+1+LOOKAHEAD)):
                if heights[k] is None or heights[k-1] is None:
                    continue
                if rej[k] or rej[k-1]:
                    continue
                if heights[k-1] - heights[k] >= DOWN_JUMP_PX:
                    drop_idx = k
                    break

            if drop_idx is None:
                i += 1
                continue

            # Mark points between up and drop as rejected IF they are clearly above baseline
            # (This prevents deleting real rises that gradually climb.)
            start = up_idx
            end = drop_idx - 1
            # Build mask: keep only segments where h is above base+ABOVE_BASE_PX
            candidates = []
            for m in range(start, end+1):
                if heights[m] is None or rej[m]:
                    continue
                if heights[m] >= base + ABOVE_BASE_PX:
                    candidates.append(m)

            if len(candidates) >= MIN_LEN:
                # Reject full contiguous span from first to last candidate
                a = candidates[0]
                b = candidates[-1]
                for m in range(a, b+1):
                    if rows[m].get("rejected","0") != "1":
                        rows[m]["rejected"] = "1"
                        marked += 1
                i = drop_idx + 1
                continue

        i += 1

    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp, CSV_PATH)
    print(f"Plateau cleanup v2 done. Newly marked rejected: {marked}")
    print("Updated CSV:", CSV_PATH)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
