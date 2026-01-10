import csv, sys, os, json
from datetime import datetime
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/starter_cam")
LOG = os.path.join(BASE, "starter_log.csv")
OUT_JSON = os.path.join(BASE, "reports", "session_latest.json")
OUT_PNG  = os.path.join(BASE, "reports", "session_latest.png")

def parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def main():
    if len(sys.argv) != 3:
        print("usage: session_report.py START_ISO END_ISO")
        return 1

    start = parse_iso(sys.argv[1])
    end   = parse_iso(sys.argv[2])
    if not start or not end:
        print("invalid start/end time")
        return 1

    rows = []
    with open(LOG, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 3:
                continue
            ts = parse_iso(r[0])
            if not ts or not (start <= ts <= end):
                continue
            try:
                y = float(r[1])
                rej = int(r[2])
            except Exception:
                continue
            rows.append((ts, y, rej))

    if not rows:
        print("no data points in range")
        return 1

    rows.sort(key=lambda r: r[0])

    # --- split series ---
    ts_list = [r[0] for r in rows]
    ys_raw  = [r[1] for r in rows]
    rejs    = [r[2] for r in rows]

    # --- convert surface_y → rise (human-intuitive) ---
    y0 = ys_raw[0]
    initial_fill_px = max(1.0, (441 - y0))
    rise_pct = [((y0 - y) / initial_fill_px) * 100.0 for y in ys_raw]

    # --- hours since start ---
    t0 = ts_list[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in ts_list]

    # --- write JSON ---
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "points": [
                    {
                        "ts": ts.isoformat(),
                        "hour": h,
                        "rise_pct": rp,
                        "rejected": rej
                    }
                    for ts, h, rp, rej in zip(ts_list, hours, rise_pct, rejs)
                ]
            },
            f,
            indent=2
        )

    # --- plot static graph ---
    plt.figure(figsize=(10, 4))
    plt.plot(hours, rise_pct, linewidth=2)
    plt.title("Session rise (%)")
    plt.xlabel("Hours since start")
    plt.ylabel("Rise (%)")
    plt.grid(True, alpha=0.3)

    # Dynamic hour ticks (1 → N, cap at 24)
    max_h = int(hours[-1])
    max_h = max(1, min(24, max_h))
    plt.xticks(list(range(1, max_h + 1)))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print("Wrote:")
    print(" ", OUT_PNG)
    print(" ", OUT_JSON)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
