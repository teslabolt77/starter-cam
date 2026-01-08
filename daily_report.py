import csv, json, os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOG = os.path.expanduser("~/starter_cam/starter_log.csv")
OUT_PNG = os.path.expanduser("~/starter_cam/reports/latest.png")
OUT_JSON = os.path.expanduser("~/starter_cam/reports/latest.json")

WINDOW_START_HOUR = 21  # 9 PM
WINDOW_END_HOUR = 12    # 12 PM

def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

def moving_avg(xs, k=7):
    if k <= 1 or len(xs) < k:
        return xs[:]
    out = []
    half = k // 2
    for i in range(len(xs)):
        a = max(0, i-half)
        b = min(len(xs), i+half+1)
        out.append(sum(xs[a:b]) / (b-a))
    return out

def compute_window(now: datetime):
    today_noon = now.replace(hour=WINDOW_END_HOUR, minute=0, second=0, microsecond=0)
    if now < today_noon:
        start = (now - timedelta(days=1)).replace(hour=WINDOW_START_HOUR, minute=0, second=0, microsecond=0)
        end = now
    else:
        end = today_noon
        start = (end - timedelta(days=1)).replace(hour=WINDOW_START_HOUR, minute=0, second=0, microsecond=0)
    return start, end

def is_rejected(row: dict) -> bool:
    try:
        return int(row.get("rejected", "0")) == 1
    except Exception:
        return False

def main():
    if not os.path.exists(LOG):
        print("No log file yet.")
        return 0

    now = datetime.now()
    start, end = compute_window(now)

    t_ok, h_ok = [], []
    t_bad, h_bad = [], []
    total = 0

    with open(LOG, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = parse_iso(row["timestamp"])
            if not (start <= ts <= end):
                continue

            total += 1
            h = float(row["height_px"])
            if is_rejected(row):
                t_bad.append(ts)
                h_bad.append(h)
            else:
                t_ok.append(ts)
                h_ok.append(h)

    if len(h_ok) < 3:
        print("Not enough accepted data in window.")
        return 0

    hs = moving_avg(h_ok, k=7)

    peak_i = max(range(len(hs)), key=lambda i: hs[i])
    peak_time = t_ok[peak_i]
    peak_height = hs[peak_i]

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Accepted data
    ax.plot(t_ok, h_ok, marker="o", linewidth=1, alpha=0.35, label="Accepted")
    ax.plot(t_ok, hs, linewidth=2, label="Smoothed")

    # Rejected points — faint red X
    if t_bad:
        ax.scatter(
            t_bad, h_bad,
            marker="x",
            s=40,
            linewidths=1.5,
            color="red",
            alpha=0.25,
            label="Rejected"
        )

    ax.axvline(peak_time, linestyle="--", linewidth=1, label="Peak")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))

    ax.set_xlim(start, end)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Starter rise • Peak ~ {peak_time.strftime('%-I:%M %p')}")
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()

    summary = {
        "window_start": start.isoformat(timespec="seconds"),
        "window_end": end.isoformat(timespec="seconds"),
        "samples_total": total,
        "samples_accepted": len(h_ok),
        "samples_rejected": len(h_bad),
        "peak_time": peak_time.isoformat(timespec="seconds"),
        "peak_height_px_smoothed": round(peak_height, 2),
        "latest_time": t_ok[-1].isoformat(timespec="seconds"),
        "latest_height_px": round(h_ok[-1], 2),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote updated graph + summary")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
