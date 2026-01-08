import os, subprocess
from datetime import datetime, timedelta

NIGHT_DIR = os.path.expanduser("~/starter_cam/photos_night")
OUT_DIR = os.path.expanduser("~/starter_cam/timelapse")

def night_window(now: datetime):
    end = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now < end:
        end = end - timedelta(days=1)
    start = (end - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
    return start, end

def main():
    now = datetime.now()
    start, end = night_window(now)

    if not os.path.isdir(NIGHT_DIR):
        print("No night photo dir.")
        return 0

    # photos are named like night_2026-01-03T22-00-00.jpg
    files = []
    for fn in os.listdir(NIGHT_DIR):
        if not (fn.startswith("night_") and fn.endswith(".jpg")):
            continue
        # parse timestamp portion
        ts = fn[len("night_"):-len(".jpg")].replace("-", ":", 2).replace("-", ":", 1)
        # safer: just rely on filename sort (ISO-like) by keeping original fn
        files.append(fn)

    files = sorted(files)
    if len(files) < 5:
        print("Not enough frames.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    list_path = os.path.join(OUT_DIR, "frames.txt")

    # concat demuxer list file
    with open(list_path, "w") as f:
        for fn in files:
            f.write(f"file '{os.path.join(NIGHT_DIR, fn)}'\n")

    date_tag = start.strftime("%Y-%m-%d")
    out_latest = os.path.join(OUT_DIR, "latest.mp4")
    out_dated = os.path.join(OUT_DIR, f"night_{date_tag}.mp4")

    # 12 fps output. Adjust -r if you want faster/slower playback.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vf", "fps=12,format=yuv420p",
        out_dated
    ]
    subprocess.run(cmd, check=False)

    # copy/overwrite latest
    try:
        subprocess.run(["cp", "-f", out_dated, out_latest], check=True)
    except Exception:
        pass

    print("Wrote:", out_dated)
    print("Wrote:", out_latest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
