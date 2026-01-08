import os, json, subprocess, sys
from datetime import datetime, timedelta

BASE = os.path.expanduser("~/starter_cam")
STATE = os.path.join(BASE, "web", "state.json")
REPORT = os.path.join(BASE, "session_report.py")
TL = os.path.join(BASE, "build_timelapse_session.py")

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def load_state():
    try:
        with open(STATE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2)

def run_cmd(cmd):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout.rstrip())
    if p.stderr:
        print(p.stderr.rstrip(), file=sys.stderr)
    print("exit:", p.returncode)
    return p.returncode

def main():
    st = load_state()
    end = now_iso()

    # Optional: limit report window to last N minutes (great for quick tests)
    last_min_env = os.environ.get("FINALIZE_LAST_MINUTES", "").strip()
    last_minutes = None
    if last_min_env.isdigit():
        last_minutes = int(last_min_env)

    start = st.get("start_time")
    start_dt = parse_iso(start) if start else None
    end_dt = parse_iso(end)

    # If start_time missing or absurdly old, fallback to last 60 minutes
    if last_minutes is not None:
        start_dt = end_dt - timedelta(minutes=last_minutes)
        start = start_dt.isoformat(timespec="seconds")
    elif (start_dt is None) or (end_dt and start_dt and (end_dt - start_dt).total_seconds() > 72*3600):
        start_dt = end_dt - timedelta(minutes=60)
        start = start_dt.isoformat(timespec="seconds")

    st["status"] = "Generating report + timelapse…"
    st["running"] = False
    st["end_time"] = end
    save_state(st)

    # Run report + timelapse with visible logs
    if start:
        run_cmd(["python3", REPORT, start, end])
    else:
        print("WARNING: No start_time; skipping report.")

    run_cmd(["python3", TL])

    st = load_state()
    st["status"] = "Completed ✅"
    st["completed"] = True
    st["running"] = False
    save_state(st)

if __name__ == "__main__":
    main()
