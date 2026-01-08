from flask import Flask, jsonify, render_template_string, redirect, send_file, make_response
import os, json, subprocess
from datetime import datetime

BASE = os.path.expanduser("~/starter_cam")
STATE_PATH = os.path.join(BASE, "web", "state.json")
LOG_CSV = os.path.join(BASE, "starter_log.csv")

REPORT_PNG = os.path.join(BASE, "reports", "session_latest.png")
TIMELAPSE  = os.path.join(BASE, "timelapse", "session_latest.mp4")
LIVE_IMG   = os.path.join(BASE, "reports", "live.jpg")

SERVICE = "starter-capture.service"

app = Flask(__name__)

def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)

def service_active():
    return subprocess.run(
        ["systemctl", "is-active", SERVICE],
        capture_output=True
    ).returncode == 0

def read_log_tail(n=600):
    pts = []
    if not os.path.exists(LOG_CSV):
        return pts
    with open(LOG_CSV) as f:
        for ln in f.readlines()[-n:]:
            try:
                ts, y, rej = ln.strip().split(",")
                pts.append({
                    "ts": ts,
                    "y": float(y),
                    "rejected": int(rej)
                })
            except Exception:
                pass
    return pts

def last_from_csv():
    pts = read_log_tail(1)
    if not pts:
        return None, None, None
    p = pts[-1]
    return p["ts"], p["y"], p["rejected"]

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Sourdough Starter Monitor</title>
<style>
body { background:#0e0e11; color:#e6e6eb; font-family:system-ui; padding:20px }
.card { background:#16161d; padding:16px; border-radius:14px; max-width:980px }
canvas { width:100%; border:1px solid #2a2a35; border-radius:10px }
.muted { color:#9a9ab0 }
button { padding:8px 12px; border-radius:10px }
</style>
</head>
<body>

<h2>Sourdough Starter Monitor</h2>
<div class="card">

<form method="POST" action="/start"><button {{ 'disabled' if running else '' }}>Start</button></form>
<form method="POST" action="/stop"><button {{ 'disabled' if not running else '' }}>Stop</button></form>

<p class="muted">
Start: {{ start_time or '—' }}<br>
Uptime: {{ uptime_min or '—' }} min<br>
Initial height: {{ initial_height_px or '—' }} px<br>
Latest height: {{ live_height or '—' }} px
</p>

<canvas id="chart" width="900" height="420"></canvas>

<p>
<a href="/graph">Static graph</a> |
<a href="/timelapse">Timelapse</a>
</p>

</div>

<script>
async function poll(){
  const pts = await fetch("/api/log").then(r=>r.json());
  draw(pts);
}

function draw(points){
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);

  if(points.length < 2) return;

  const t0 = new Date(points[0].ts).getTime();
  const ys = points.map(p=>p.y);
  const y0 = ys[0];
  const rise = ys.map(v => y0 - v);

  const maxY = Math.max(...rise);
  const minY = Math.min(...rise);

  const ml=60, mr=20, mt=20, mb=40;
  const iw=c.width-ml-mr, ih=c.height-mt-mb;

  function x(i){ return ml + (i/(points.length-1))*iw }
  function y(v){ return mt + (1-(v-minY)/(maxY-minY||1))*ih }

  // axes
  ctx.strokeStyle="#333";
  ctx.beginPath();
  ctx.moveTo(ml,mt);
  ctx.lineTo(ml,mt+ih);
  ctx.lineTo(ml+iw,mt+ih);
  ctx.stroke();

  // hour ticks
  const hours = Math.floor((new Date(points.at(-1).ts)-new Date(points[0].ts))/3600000);
  for(let h=1; h<=hours; h++){
    const px = ml + (h/(hours||1))*iw;
    ctx.fillText(h+"h", px-8, mt+ih+20);
    ctx.beginPath();
    ctx.moveTo(px,mt);
    ctx.lineTo(px,mt+ih);
    ctx.stroke();
  }

  // line
  ctx.strokeStyle="#ffb86b";
  ctx.beginPath();
  rise.forEach((v,i)=>{
    if(i===0) ctx.moveTo(x(i),y(v));
    else ctx.lineTo(x(i),y(v));
  });
  ctx.stroke();
}

poll();
setInterval(poll,5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    st = load_state()
    ts,h,_ = last_from_csv()
    return render_template_string(
        HTML,
        running=service_active(),
        start_time=st.get("start_time"),
        uptime_min=st.get("uptime_min"),
        initial_height_px=st.get("initial_height_px"),
        live_height=h
    )

@app.route("/api/log")
def api_log():
    return no_cache(make_response(jsonify(read_log_tail(600))))

@app.route("/start", methods=["POST"])
def start():
    st = load_state()
    st["start_time"] = datetime.now().isoformat(timespec="seconds")
    st["uptime_min"] = 0
    save_state(st)
    subprocess.run(["sudo","systemctl","start",SERVICE])
    return redirect("/")

@app.route("/stop", methods=["POST"])
def stop():
    subprocess.run(["sudo","systemctl","stop",SERVICE])
    return redirect("/")

@app.route("/graph")
def graph():
    return send_file(REPORT_PNG)

@app.route("/timelapse")
def timelapse():
    return send_file(TIMELAPSE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
