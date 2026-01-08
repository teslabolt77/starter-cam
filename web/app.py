from flask import Flask, jsonify, render_template_string, redirect, send_file, make_response
import os, json, subprocess, glob
from datetime import datetime

BASE = os.path.expanduser("~/starter_cam")
STATE_PATH  = os.path.join(BASE, "web", "state.json")

REPORT_JSON = os.path.join(BASE, "reports", "session_latest.json")
REPORT_PNG  = os.path.join(BASE, "reports", "session_latest.png")
TIMELAPSE   = os.path.join(BASE, "timelapse", "session_latest.mp4")

# "live" image is optional — we'll fall back to newest overlay frame automatically
LIVE_IMG    = os.path.join(BASE, "reports", "live.jpg")
PHOTOS_DIR  = os.path.join(BASE, "photos_run")

LOG_CSV = os.path.join(BASE, "starter_log.csv")
CAPTURE_SCRIPT = os.path.join(BASE, "starter_capture.py")
FINALIZE = os.path.join(BASE, "finalize_session.py")

SERVICE = "starter-capture.service"

app = Flask(__name__)

def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)

def service_active():
    r = subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True)
    return r.returncode == 0

def read_log_tail(n=600):
    pts = []
    if not os.path.exists(LOG_CSV):
        return pts
    try:
        with open(LOG_CSV, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for ln in lines[-int(n):]:
            parts = ln.split(",")
            if len(parts) < 3:
                continue
            ts, y, rej = parts[0], parts[1], parts[2]
            try:
                datetime.fromisoformat(ts)
                pts.append({"ts": ts, "y": float(y), "rejected": int(rej)})
            except Exception:
                continue
    except Exception:
        pass
    return pts

def last_from_csv():
    pts = read_log_tail(1)
    if not pts:
        return None, None, None
    p = pts[-1]
    return p["ts"], p["y"], p["rejected"]

def newest_overlay_path():
    # Prefer LIVE_IMG if it exists; otherwise use newest run_overlay_*.jpg from photos_run
    if os.path.exists(LIVE_IMG):
        return LIVE_IMG
    pats = [
        os.path.join(PHOTOS_DIR, "run_overlay_*.jpg"),
        os.path.join(PHOTOS_DIR, "*.jpg"),
    ]
    cand = []
    for pat in pats:
        cand.extend(glob.glob(pat))
    if not cand:
        return None
    return max(cand, key=os.path.getmtime)

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def mtime_tag(path):
    try:
        return str(int(os.path.getmtime(path)))
    except Exception:
        return "0"

def wipe_session_artifacts():
    # wipe graph data + outputs so a new run doesn't show old stuff
    try:
        open(LOG_CSV, "w").close()
    except Exception:
        pass
    for p in [REPORT_JSON, REPORT_PNG, TIMELAPSE, LIVE_IMG]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def test_capture_live():
    tmpdir = os.path.join(BASE, "reports", "live_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    subprocess.run(
        ["python3", CAPTURE_SCRIPT, "--save-image", "--image-dir", tmpdir, "--prefix", "live"],
        check=False
    )

    jpgs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".jpg")]
    if not jpgs:
        return False
    newest = max(jpgs, key=os.path.getmtime)
    os.makedirs(os.path.dirname(LIVE_IMG), exist_ok=True)
    subprocess.run(["cp", "-f", newest, LIVE_IMG], check=False)

    for f in jpgs:
        try:
            os.remove(f)
        except Exception:
            pass
    return True

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sourdough Starter Monitor</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin:0; padding:20px; background:#0e0e11; color:#e6e6eb; }
    h2 { margin:0 0 12px 0; font-weight:600; }
    .card { background:#16161d; border:1px solid #2a2a35; padding:16px; border-radius:14px; max-width: 980px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
    .row { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0; align-items:center; }
    button { padding:10px 14px; border-radius:12px; border:1px solid #2a2a35; background:#222232; color:#e6e6eb; cursor:pointer; }
    button:hover { filter: brightness(1.1); }
    button:disabled { opacity:0.5; cursor:not-allowed; }
    a { color:#7aa2ff; text-decoration:none; }
    a:hover { text-decoration:underline; }
    img { max-width:100%; border-radius:10px; border:1px solid #2a2a35; background:#0e0e11; }
    .muted { color:#9a9ab0; font-size:0.95em; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #2a2a35; background:#111118; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    @media(max-width: 900px){ .grid{ grid-template-columns:1fr; } }
    canvas { width:100%; height:auto; border-radius:10px; border:1px solid #2a2a35; background:#0e0e11; }
    .small { font-size: 0.9em; color:#9a9ab0; }
  </style>
</head>
<body>
  <h2>Sourdough Starter Monitor</h2>
  <div class="card">
    <div class="row">
      <form method="POST" action="/start"><button {{ 'disabled' if running else '' }}>Start captures</button></form>
      <form method="POST" action="/stop"><button {{ 'disabled' if not running else '' }}>Stop + build outputs</button></form>
      <form method="POST" action="/test"><button>Test capture live</button></form>
      <span class="pill" id="statuspill">{{ status }}</span>
    </div>

    <div class="muted">
      Start: <span id="start_time">{{ start_time or '—' }}</span><br>
      Last capture: <span id="last_capture">{{ last_capture or '—' }}</span><br>
      Heartbeat: <span id="heartbeat">{{ heartbeat or '—' }}</span><br>
      Uptime: <span id="uptime">{{ uptime_min if uptime_min is not none else '—' }}</span> min<br>
      End: <span id="end_time">{{ end_time or '—' }}</span><br>
      Interval: <span id="interval_label">{{ interval_label }}</span>
    </div>

    <div class="grid" style="margin-top:14px;">
      <div>
        <b>Live preview (auto):</b><br>
        <img id="liveimg" src="/live?v={{ live_tag }}" onerror="this.style.display='none';document.getElementById('nol').style.display='block';">
        <div id="nol" class="muted" style="display:none;">No image yet — start captures or click “Test capture live”.</div>
        <div class="muted" style="margin-top:8px;">
          Latest height: <span id="live_height">{{ live_height or '—' }}</span> px
          {% if live_rej is not none %}(rejected=<span id="live_rej">{{ live_rej }}</span>){% endif %}<br>
          Latest time: <span id="live_ts">{{ live_ts or '—' }}</span>
        </div>
      </div>

      <div>
        <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
          <b>Session graph (live): Rise (px)</b>
          <span class="small">updates every 5s</span>
        </div>
        <canvas id="chart" width="900" height="420"></canvas>
        <div class="muted" id="nog" style="display:none;">No data yet.</div>

        <div style="margin-top:14px;">
          <b>Session timelapse:</b>
          <a href="/timelapse?v={{ tl_tag }}">session_latest.mp4</a><br>
          <b>Session summary:</b> <a href="/api/summary">/api/summary</a><br>
          <b>Static graph (on finalize):</b> <a href="/graph?v={{ graph_tag }}">session_latest.png</a>
        </div>
      </div>
    </div>
  </div>

<script>
function pad2(n){ return (n<10?'0':'')+n; }

function drawChart(points){
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = "#0e0e11";
  ctx.fillRect(0,0,W,H);

  if(!points || points.length < 2){
    document.getElementById("nog").style.display = "block";
    return;
  }
  document.getElementById("nog").style.display = "none";

  // margins
  const ml=55, mr=15, mt=18, mb=45;
  const iw = W-ml-mr, ih = H-mt-mb;

  // Establish initial height from first NON-rejected point
  let initY = null;
  for(const p of points){
    if(!p.rejected && typeof p.y === "number"){
      initY = p.y;
      break;
    }
  }
  if(initY === null){
    initY = points[0].y;
  }

  // Convert to "rise" so rising starter = positive numbers (because y decreases as it rises)
  const rises = points.map(p => ({
    ts: p.ts,
    r: (typeof p.y === "number") ? (initY - p.y) : null,
    rejected: p.rejected
  }));

  // range
  let ymin=Infinity, ymax=-Infinity;
  for(const p of rises){
    if(typeof p.r !== "number") continue;
    ymin = Math.min(ymin, p.r);
    ymax = Math.max(ymax, p.r);
  }
  if(!isFinite(ymin) || !isFinite(ymax)) return;

  // padding
  const pad = Math.max(5, (ymax - ymin)*0.08);
  ymin -= pad; ymax += pad;

  const t0 = new Date(rises[0].ts).getTime();
  const t1 = new Date(rises[rises.length-1].ts).getTime();
  const dt = Math.max(1, t1 - t0);

  function xFor(ts){
    const t = new Date(ts).getTime();
    return ml + ((t - t0) / dt) * iw;
  }
  function yFor(v){
    return mt + (1 - (v - ymin)/(ymax - ymin)) * ih;
  }

  // axes
  ctx.strokeStyle = "#2a2a35";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ml, mt);
  ctx.lineTo(ml, mt+ih);
  ctx.lineTo(ml+iw, mt+ih);
  ctx.stroke();

  // y ticks
  ctx.fillStyle = "#9a9ab0";
  ctx.font = "12px system-ui";
  const ticks = 5;
  for(let i=0;i<=ticks;i++){
    const vv = ymin + (i/ticks)*(ymax-ymin);
    const py = yFor(vv);
    ctx.strokeStyle = "#1f1f28";
    ctx.beginPath();
    ctx.moveTo(ml, py);
    ctx.lineTo(ml+iw, py);
    ctx.stroke();
    ctx.fillText(vv.toFixed(0), 8, py+4);
  }

  // X axis: hour marks (0h, 1h, 2h...) like the static graph
  const hourMs = 3600*1000;
  const maxHours = Math.max(1, Math.ceil(dt / hourMs));
  ctx.fillStyle = "#9a9ab0";
  for(let h=0; h<=maxHours; h++){
    const tx = t0 + h*hourMs;
    const px = ml + ((tx - t0) / dt) * iw;

    // grid line
    ctx.strokeStyle = "#1f1f28";
    ctx.beginPath();
    ctx.moveTo(px, mt);
    ctx.lineTo(px, mt+ih);
    ctx.stroke();

    // label
    const lab = h + "h";
    ctx.fillText(lab, px-10, mt+ih+28);
  }
  ctx.fillStyle = "#9a9ab0";
  ctx.font = "11px system-ui";
  ctx.fillText("hours since start", ml + iw/2 - 45, mt+ih+42);

  // Baseline (initial height): rise = 0
  const y0 = yFor(0);
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(ml, y0);
  ctx.lineTo(ml + iw, y0);
  ctx.stroke();
  ctx.setLineDash([]);

  // line (accepted only)
  ctx.strokeStyle = "#ffb86b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started=false;
  for(const p of rises){
    if(p.rejected) continue;
    const px = xFor(p.ts);
    const py = yFor(p.r);
    if(!started){ ctx.moveTo(px,py); started=true; }
    else ctx.lineTo(px,py);
  }
  ctx.stroke();

  // points
  for(const p of rises){
    const px = xFor(p.ts);
    const py = yFor(p.r);
    ctx.fillStyle = p.rejected ? "#7aa2ff" : "#e6e6eb";
    ctx.globalAlpha = p.rejected ? 0.55 : 0.9;
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1.0;

  // title
  ctx.fillStyle = "#e6e6eb";
  ctx.font = "14px system-ui";
  ctx.fillText("Rise (px)", 8, 14);
}

async function poll(){
  try{
    const r = await fetch("/api/log?n=600", {cache:"no-store"});
    const pts = await r.json();
    drawChart(pts);

    const s = await fetch("/api/state", {cache:"no-store"}).then(x=>x.json());
    if(s){
      const set = (id,val)=>{ const el=document.getElementById(id); if(el) el.textContent = (val===null||val===undefined||val==="") ? "—" : val; };
      set("statuspill", s.status);
      set("start_time", s.start_time);
      set("last_capture", s.last_capture);
      set("heartbeat", s.heartbeat);
      set("uptime", s.uptime_min);
      set("end_time", s.end_time);
      set("interval_label", s.interval_label);

      set("live_ts", s.live_ts);
      set("live_height", s.live_height);
      if(document.getElementById("live_rej")) set("live_rej", s.live_rej);

      // bust-cache live image every poll
      const live = document.getElementById("liveimg");
      if(live){
        live.src = "/live?v=" + Date.now();
      }
    }
  }catch(e){
    // ignore
  }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    st = load_state()
    running = service_active()
    st["running"] = running
    save_state(st)

    ts, h, rej = last_from_csv()

    interval_label = st.get("interval_label") or ("1 minute (test)" if "1 minute" in (st.get("status","").lower()) else "service")
    live_path = newest_overlay_path()
    live_tag = mtime_tag(live_path) if live_path else "0"

    return render_template_string(
        HTML,
        initial_height=st.get("initial_height_px"),
        running=running,
        status=st.get("status", "Idle"),
        start_time=st.get("start_time"),
        last_capture=st.get("last_capture"),
        end_time=st.get("end_time"),
        heartbeat=st.get("heartbeat"),
        uptime_min=st.get("uptime_min"),
        live_ts=ts,
        live_height=h,
        live_rej=rej if ts else None,
        interval_label=interval_label,
        graph_tag=mtime_tag(REPORT_PNG),
        tl_tag=mtime_tag(TIMELAPSE),
        live_tag=live_tag,
    )

@app.route("/api/state")
def api_state():
    st = load_state()
    running = service_active()
    ts, h, rej = last_from_csv()

    interval_label = st.get("interval_label") or "service"
    out = {
        
        \"initial_height\": st.get(\"initial_height_px\"),"running": running,
        "status": st.get("status", "Idle"),
        "start_time": st.get("start_time"),
        "last_capture": st.get("last_capture"),
        "heartbeat": st.get("heartbeat"),
        "uptime_min": st.get("uptime_min"),
        "end_time": st.get("end_time"),
        "interval_label": interval_label,
        "live_ts": ts,
        "live_height": h,
        "live_rej": rej if ts else None,
    }
    return no_cache(make_response(jsonify(out)))

@app.route("/api/log")
def api_log():
    try:
        from flask import request
        n = int(request.args.get("n", "600"))
        n = max(10, min(n, 5000))
    except Exception:
        n = 600
    pts = read_log_tail(n)
    resp = make_response(jsonify(pts))
    return no_cache(resp)

@app.route("/start", methods=["POST"])
def start():
    wipe_session_artifacts()
    st = load_state()
    now = datetime.now().isoformat(timespec="seconds")
    st["status"] = "Running captures"
    st["completed"] = False
    st["end_time"] = None
    st["start_time"] = now
    st["last_capture"] = None
    save_state(st)
    subprocess.run(["sudo", "systemctl", "start", SERVICE], check=False)
    return redirect("/")

@app.route("/stop", methods=["POST"])
def stop():
    st = load_state()
    st["status"] = "Stopping…"
    save_state(st)
    subprocess.run(["sudo", "systemctl", "stop", SERVICE], check=False)
    subprocess.run(["python3", FINALIZE], check=False)
    return redirect("/")

@app.route("/test", methods=["POST"])
def test():
    ok = test_capture_live()
    st = load_state()
    st["status"] = "Test capture complete ✅" if ok else "Test capture failed ❌"
    save_state(st)
    return redirect("/")

@app.route("/api/summary")
def summary():
    if not os.path.exists(REPORT_JSON):
        return jsonify({"error":"No session report yet (stop the run first)."}), 404
    with open(REPORT_JSON) as f:
        return jsonify(json.load(f))

@app.route("/graph")
def graph():
    if not os.path.exists(REPORT_PNG):
        return "No session graph yet", 404
    resp = make_response(send_file(REPORT_PNG, mimetype="image/png", conditional=False))
    return no_cache(resp)

@app.route("/timelapse")
def timelapse():
    if not os.path.exists(TIMELAPSE):
        return "No session timelapse yet", 404
    resp = make_response(send_file(TIMELAPSE, mimetype="video/mp4", conditional=False))
    return no_cache(resp)

@app.route("/live")
def live():
    live_path = newest_overlay_path()
    if not live_path or not os.path.exists(live_path):
        return "No live preview yet", 404
    resp = make_response(send_file(live_path, mimetype="image/jpeg", conditional=False))
    return no_cache(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
