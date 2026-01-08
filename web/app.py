from flask import Flask, jsonify, render_template_string, redirect, send_file, make_response
import os, json, subprocess
from datetime import datetime

BASE = os.path.expanduser("~/starter_cam")
STATE_PATH  = os.path.join(BASE, "web", "state.json")

REPORT_JSON = os.path.join(BASE, "reports", "session_latest.json")
REPORT_PNG  = os.path.join(BASE, "reports", "session_latest.png")
TIMELAPSE   = os.path.join(BASE, "timelapse", "session_latest.mp4")
LIVE_IMG    = os.path.join(BASE, "reports", "live.jpg")

LOG_CSV = os.path.join(BASE, "starter_log.csv")
CAPTURE_SCRIPT = os.path.join(BASE, "starter_capture.py")
FINALIZE = os.path.join(BASE, "finalize_session.py")

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

def last_height_from_csv():
    if not os.path.exists(LOG_CSV):
        return None, None, None
    try:
        with open(LOG_CSV) as f:
            lines = [l.strip() for l in f if l.strip()]
        ts, h, r = lines[-1].split(",")[:3]
        return ts, h, r
    except Exception:
        return None, None, None

def read_log_tail(n=300):
    pts = []
    if not os.path.exists(LOG_CSV):
        return pts
    with open(LOG_CSV) as f:
        for ln in f.readlines()[-n:]:
            try:
                ts, y, r = ln.strip().split(",")[:3]
                datetime.fromisoformat(ts)
                pts.append({"ts": ts, "y": float(y), "rejected": int(r)})
            except Exception:
                pass
    return pts

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

def mtime_tag(p):
    try:
        return str(int(os.path.getmtime(p)))
    except Exception:
        return "0"

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Sourdough Starter Monitor</title>
<style>
body{background:#0e0e11;color:#e6e6eb;font-family:system-ui;padding:20px}
.card{background:#16161d;border-radius:14px;padding:16px;max-width:980px}
canvas{background:#0e0e11;border:1px solid #2a2a35;border-radius:10px}
.muted{color:#9a9ab0}
</style>
</head>
<body>
<h2>Sourdough Starter Monitor</h2>
<div class="card">
<b>Live preview:</b><br>
<img src="/live?v={{ live_tag }}" width="420"><br>
<div class="muted">
Initial height: {{ initial_height or "—" }} px<br>
Latest height: {{ live_height or "—" }} px<br>
Latest time: {{ live_ts or "—" }}
</div>

<br><b>Session graph (live):</b>
<canvas id="chart" width="900" height="420"></canvas>

<br>
<a href="/timelapse?v={{ tl_tag }}">session_latest.mp4</a><br>
<a href="/graph?v={{ graph_tag }}">session_latest.png</a>
</div>

<script>
function drawChart(points){
 const c=document.getElementById("chart");
 const ctx=c.getContext("2d");
 ctx.clearRect(0,0,c.width,c.height);

 if(points.length<2) return;

 const ml=60,mr=20,mt=20,mb=40;
 const iw=c.width-ml-mr, ih=c.height-mt-mb;

 let ymin=1e9,ymax=-1e9;
 points.forEach(p=>{ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y)});
 let pad=(ymax-ymin)*0.1;
 ymin-=pad; ymax+=pad;

 const t0=new Date(points[0].ts).getTime();
 const t1=new Date(points.at(-1).ts).getTime();
 const hrs=(t1-t0)/3600000;

 const xFor=t=>ml+((new Date(t).getTime()-t0)/3600000)/hrs*iw;
 const yFor=y=>mt+(1-(y-ymin)/(ymax-ymin))*ih;

 ctx.strokeStyle="#2a2a35";
 ctx.beginPath();
 ctx.moveTo(ml,mt);
 ctx.lineTo(ml,mt+ih);
 ctx.lineTo(ml+iw,mt+ih);
 ctx.stroke();

 ctx.fillStyle="#9a9ab0";
 for(let h=0;h<=Math.floor(hrs);h++){
   let x=ml+(h/hrs)*iw;
   ctx.fillText("h "+h,x-8,mt+ih+20);
 }

 ctx.strokeStyle="#ffb86b";
 ctx.beginPath();
 let s=false;
 points.forEach(p=>{
   if(p.rejected) return;
   let x=xFor(p.ts), y=yFor(p.y);
   if(!s){ctx.moveTo(x,y);s=true}else ctx.lineTo(x,y);
 });
 ctx.stroke();
}

async function poll(){
 const pts=await fetch("/api/log?n=600").then(r=>r.json());
 drawChart(pts);
}
poll(); setInterval(poll,5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    st = load_state()
    ts,h,_ = last_height_from_csv()
    return render_template_string(
        HTML,
        live_ts=ts,
        live_height=h,
        initial_height=st.get("initial_height_px"),
        live_tag=mtime_tag(LIVE_IMG),
        tl_tag=mtime_tag(TIMELAPSE),
        graph_tag=mtime_tag(REPORT_PNG),
    )

@app.route("/api/log")
def api_log():
    return no_cache(make_response(jsonify(read_log_tail(600))))

@app.route("/graph")
def graph():
    return no_cache(send_file(REPORT_PNG))

@app.route("/timelapse")
def timelapse():
    return no_cache(send_file(TIMELAPSE))

@app.route("/live")
def live():
    return no_cache(send_file(LIVE_IMG))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)

