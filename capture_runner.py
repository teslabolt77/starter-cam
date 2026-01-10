#!/usr/bin/env python3
import os, time, json, subprocess, re, signal
from datetime import datetime

BASE = os.path.expanduser("~/starter_cam")
WEB_STATE  = os.path.join(BASE, "web", "state.json")
PHOTOS_RUN = os.path.join(BASE, "photos_run")
CAPTURE    = os.path.join(BASE, "starter_capture.py")
LOG_CSV    = os.path.join(BASE, "starter_log.csv")

# Default interval = 15 minutes (900s). Override with:
#   export CAPTURE_INTERVAL_S=60
CAPTURE_INTERVAL_S = int(os.environ.get("CAPTURE_INTERVAL_S", "60"))

# Parse line emitted by starter_capture.py:
# SURFACE_Y 292 conf=0.70 ROI[292:338] JAR[91:395] [REJECTED]
RE_PARSE = re.compile(
    r"SURFACE_Y\s+(?P<y>\d+)\s+conf=(?P<conf>[-+]?\d+(?:\.\d+)?)\s+ROI\[(?P<roi1>\d+):(?P<roi2>\d+)\]\s+JAR\[(?P<jt>\d+):(?P<jb>\d+)\](?P<rej>.*REJECTED.*)?",
    re.IGNORECASE
)

stop_flag = False

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

def ensure_csv_header():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as f:
            f.write("timestamp,surface_y_px,conf,jar_top_px,jar_bot_px,rejected\n")

def append_log(ts, y, conf, jt, jb, rejected):
    ensure_csv_header()
    with open(LOG_CSV, "a") as f:
        f.write(f"{ts},{y},{conf:.4f},{jt},{jb},{1 if rejected else 0}\n")

def capture_once():
    """
    Runs starter_capture.py once.
    Returns:
      (y, conf, jt, jb, rejected, stdout_text)
    or:
      (None, None, None, None, True, stdout_text) on parse failure
    """
    os.makedirs(PHOTOS_RUN, exist_ok=True)

    # IMPORTANT:
    # We do NOT set jar fractions here anymore; starter_capture.py should auto-detect jar every frame.
    # We keep only basic run args.
    cmd = [
        "python3", CAPTURE,
        "--width", "640", "--height", "480",
        "--save-image",
        "--image-dir", PHOTOS_RUN,
        "--prefix", "run"
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = RE_PARSE.search(out)

    if not m:
        return None, None, None, None, True, out

    y = int(m.group("y"))
    conf = float(m.group("conf"))
    jt = int(m.group("jt"))
    jb = int(m.group("jb"))
    rejected = bool(m.group("rej"))
    return y, conf, jt, jb, rejected, out

def handle_signal(signum, frame):
    global stop_flag
    stop_flag = True

def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    start_epoch = time.time()

    # Session reset
    st = {
        "running": True,
        "status": f"Running captures ({CAPTURE_INTERVAL_S//60} min)",
        "start_time": now_iso(),
        "end_time": None,
        "completed": False,
        "uptime_min": 0,
        "initial_height_px": None,

        # last values
        "last_capture": None,
        "last_height_px": None,
        "last_conf": None,
        "last_rejected": None,
        "last_ok": False,

        # jar values (px) from the detector
        "jar_top_px": None,
        "jar_bot_px": None,

        "heartbeat": now_iso(),
    }
    save_state(st)

    while not stop_flag:
        ts = now_iso()

        y, conf, jt, jb, rejected, raw = capture_once()

        st = load_state()
        st["heartbeat"] = now_iso()
        st["uptime_min"] = int((time.time() - start_epoch) // 60)
        st["status"] = f"Running captures ({CAPTURE_INTERVAL_S//60} min)"
        st["running"] = True
        st["completed"] = False
        st["end_time"] = None

        if y is None:
            # Capture ran but parsing failed
            st.update({
                "last_capture": ts,
                "last_ok": False,
                "last_rejected": 1,
            })
            # Keep prior last_height_px / last_conf if you want continuity
            save_state(st)
        else:
            if st.get("initial_height_px") is None and (not rejected):
                st["initial_height_px"] = y

            append_log(ts, y, conf, jt, jb, rejected)

            st.update({
                "last_capture": ts,
                "last_height_px": y,
                "last_conf": conf,
                "last_rejected": 1 if rejected else 0,
                "last_ok": True if not rejected else False,
                "jar_top_px": jt,
                "jar_bot_px": jb,
            })
            save_state(st)

        # Sleep until next capture (but allow fast exit)
        for _ in range(CAPTURE_INTERVAL_S):
            if stop_flag:
                break
            time.sleep(1)

    # graceful shutdown state
    st = load_state()
    st.update({
        "running": False,
        "completed": True,
        "status": "Stopped",
        "end_time": now_iso(),
        "heartbeat": now_iso(),
    })
    save_state(st)

if __name__ == "__main__":
    main()
