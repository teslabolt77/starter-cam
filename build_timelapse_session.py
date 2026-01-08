import os, subprocess, re, json
from datetime import datetime

IN_DIR   = os.path.expanduser("~/starter_cam/photos_run")
OUT_MP4  = os.path.expanduser("~/starter_cam/timelapse/session_latest.mp4")
LIST_TXT = os.path.expanduser("~/starter_cam/timelapse/session_frames.txt")
STATE    = os.path.expanduser("~/starter_cam/web/state.json")

FPS = 24

# run_overlay_YYYYMMDD_HHMMSS.jpg
RE = re.compile(r"^run_overlay_(\d{8})_(\d{6})\.jpg$")

def parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def parse_fn_time(fn):
    m = RE.match(fn)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return dt

def load_window():
    """
    Use start/end from web/state.json if present.
    If end_time missing OR session still running, use newest overlay frame time as end.
    """
    try:
        st = json.load(open(STATE))
    except Exception:
        st = {}

    start = parse_iso(st.get("start_time", "")) or datetime.min
    end_state = parse_iso(st.get("end_time", ""))

    # Determine if session is running
    running = bool(st.get("running", False)) and not bool(st.get("completed", False))

    # If no end_time, or running session, prefer the newest frame time as end
    if end_state is None or running:
        newest = None
        if os.path.isdir(IN_DIR):
            for fn in os.listdir(IN_DIR):
                t = parse_fn_time(fn)
                if t is None:
                    continue
                if newest is None or t > newest:
                    newest = t
        end = newest or datetime.now()
    else:
        end = end_state

    if end < start:
        end = datetime.now()
    return start, end

def main():
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    if not os.path.isdir(IN_DIR):
        print("No photos_run dir.")
        return 0

    start, end = load_window()

    # Only overlay frames, only within session window
    candidates = []
    for fn in os.listdir(IN_DIR):
        t = parse_fn_time(fn)
        if t is None:
            continue
        if start <= t <= end:
            candidates.append((t, fn))

    candidates.sort(key=lambda x: x[0])
    files = [fn for _, fn in candidates]

    # Fallback: if the web/state.json window is too short (common when starting),
    # build from the most recent overlay frames instead of exiting.
    if len(files) < 5:
        all_overlays = []
        for fn in os.listdir(IN_DIR):
            t = parse_fn_time(fn)
            if t is None:
                continue
            all_overlays.append((t, fn))
        all_overlays.sort(key=lambda x: x[0])
        files = [fn for _, fn in all_overlays][-2000:]  # cap for safety

    if len(files) < 5:
        print(f"Not enough overlay frames in window {start.isoformat()} -> {end.isoformat()} (found {len(files)}).")
        return 0

    with open(LIST_TXT, "w") as f:
        for fn in files:
            f.write(f"file '{os.path.join(IN_DIR, fn)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", LIST_TXT,
        "-vf", f"fps={FPS},format=yuv420p",
        OUT_MP4
    ]
    subprocess.run(cmd, check=False)
    print("Wrote timelapse:", OUT_MP4)
    print("Frames used:", len(files), "| first:", files[0], "| last:", files[-1])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
