#!/usr/bin/env python3
import os, time, json, subprocess, re
from datetime import datetime

BASE = os.path.expanduser("~/starter_cam")
WEB_STATE = os.path.join(BASE, "web", "state.json")
PHOTOS_RUN = os.path.join(BASE, "photos_run")
CAPTURE = os.path.join(BASE, "starter_capture.py")
LOG_CSV = os.path.join(BASE, "starter_log.csv")

CAPTURE_INTERVAL_S = 60
HEARTBEAT_S = 5

RE_PARSE = re.compile(
    r"SURFACE_Y\s+(?P<y>\d+)\s+conf=(?P<conf>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE
)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_state():
    try:
        with open(WEB_STATE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(os.path.dirname(WEB_STATE), exist_ok=True)
    with open(WEB_STATE, "w") as f:
        json.dump(st, f, indent=2)

def append_log(ts, y):
    with open(LOG_CSV, "a") as f:
        f.write(f"{ts},{y},0\n")

def capture_once():
    os.makedirs(PHOTOS_RUN, exist_ok=True)
    proc = subprocess.run(
        ["python3", CAPTURE,
         "--width", "640", "--height", "480",
         "--save-image", "--image-dir", PHOTOS_RUN, "--prefix", "run"],
        text=True,
        capture_output=True
    )
    m = RE_PARSE.search((proc.stdout or "") + (proc.stderr or ""))
    if not m:
        return None, None
    return int(m.group("y")), float(m.group("conf"))

def main():
    start_epoch = time.time()

    # NEW SESSION RESET
    st = {
        "running": True,
        "status": "Running captures (1 min)",
        "start_time": now_iso(),
        "end_time": None,
        "completed": False,
        "uptime_min": 0,
        "initial_height_px": None,
    }
    save_state(st)

    while True:
        ts = now_iso()
        y, conf = capture_once()

        st = load_state()

        if y is not None:
            if st.get("initial_height_px") is None:
                st["initial_height_px"] = y

            append_log(ts, y)

            st.update({
                "last_capture": ts,
                "last_height_px": y,
                "last_conf": conf,
                "last_rejected": 0,
                "last_ok": True,
            })

        st.update({
            "heartbeat": now_iso(),
            "uptime_min": int((time.time() - start_epoch) // 60),
            "status": "Running captures (1 min)",
        })

        save_state(st)
        time.sleep(CAPTURE_INTERVAL_S)

if __name__ == "__main__":
    main()

