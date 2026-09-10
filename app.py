import streamlit as st
import cv2
import time
import numpy as np
import json
import os
from datetime import date, timedelta, datetime

st.set_page_config(page_title="Focus Companion 🌿", layout="wide")

DATA_FILE = "data.json"

# ---------------- DATA ----------------
def load_data():
    default = {"history": {}, "streak": 0, "garden": []}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
        except:
            data = default
    else:
        data = default

    for d in data["history"]:
        entry = data["history"][d]
        entry["focus_minutes"] = entry.get("focus_minutes", 0)
        entry["sessions"] = entry.get("sessions", 1)
        sessions = entry["sessions"] if entry["sessions"] > 0 else 1
        entry["avg_session"] = entry.get("avg_session", entry["focus_minutes"] / sessions)
        entry["hourly_focus"] = entry.get("hourly_focus", {})
    return data

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ---------------- SESSION STATE ----------------
data = load_data()
today = str(date.today())

defaults = {
    "running": False,
    "focus_time": 0,
    "session_started_at": None,
    "last_time": time.time(),
    "cap": None,
    "last_seen_time": time.time(),
    "mode": "normal",

    # Pomodoro
    "pomodoro_state": "focus",
    "pomodoro_start": time.time(),
    "pomodoro_duration": 25*60,
    "break_duration": 5*60,

    # burnout
    "cooldown_until": 0,
    "last_active_time": time.time()
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- FACE DETECTION ----------------
@st.cache_resource
def load_cascade():
    # Avoid relying on cv2.data, which is absent in some Cloud OpenCV builds.
    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")
    return cascade

face_cascade = load_cascade()

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60,60))
    return len(faces) > 0

# ---------------- UTIL ----------------
def get_plant(m):
    if m < 5: return "🌰"
    elif m < 15: return "🌱"
    elif m < 30: return "🌿"
    elif m < 60: return "🌳"
    return "🌲"

def compute_streak(history, today_str):
    d = date.fromisoformat(today_str)
    streak = 0
    while True:
        key = str(d)
        if key in history and history[key].get("focus_minutes", 0) > 0:
            streak += 1
            d -= timedelta(days=1)
        else:
            break
    return streak

def get_break_activity():
    return np.random.choice([
        "👀 20-20-20 rule",
        "🧍 Stretch",
        "💧 Drink water",
        "🚶 Walk",
        "🫁 Deep breathing"
    ])

# ---------------- ANALYZER (FULL RESTORED) ----------------
def analyze_focus_patterns(history):
    if len(history) < 3:
        return "LOW", "Not enough data yet.", {}

    dates = sorted(history.keys())
    focus = np.array([history[d]["focus_minutes"] for d in dates])
    sessions = np.array([history[d]["sessions"] for d in dates])
    avg_session = np.array([history[d]["avg_session"] for d in dates])

    baseline = np.mean(focus[:-1])
    std = np.std(focus) + 1e-5
    consistency = 1 / (1 + std/10)
    trend = np.polyfit(range(len(focus)), focus, 1)[0]

    recent_focus = focus[-1]
    recent_sessions = sessions[-1]
    recent_avg = avg_session[-1]

    focus_quality = recent_avg / (recent_focus + 1e-5)
    fragmentation = recent_sessions / (recent_focus + 1e-5)
    z = (recent_focus - baseline)/std

    score = 0
    if z > 1.5: score += 2
    if fragmentation > 0.1: score += 1
    if focus_quality < 0.3: score += 1
    if consistency < 0.3: score += 1

    if score >= 4:
        level = "HIGH RISK (PLEASE USE THE PREVENTION MEASURES!!!)"
    elif score >= 2:
        level = "MEDIUM RISK (THERE IS STILL SOME TIME TO CHANGE.)"
    else:
        level = "LOW RISK (YOU ARE SAFE!)"

    insights = {
        "Consistency": round(consistency,2),
        "Trend": round(trend,2),
        "Focus Quality": round(focus_quality,2),
        "Fragmentation": round(fragmentation,2)
    }

    return level, "Analysis complete", insights

def generate_human_insights(insights):
    msgs = []
    if insights.get("Focus Quality",1) < 0.3:
        msgs.append("😵 Sessions too fragmented")
    if insights.get("Fragmentation",0) > 0.1:
        msgs.append("📱 Too much task switching")
    if insights.get("Trend",0) > 5:
        msgs.append("🚀 Strong improvement")
    if not msgs:
        msgs.append("✨ Stable focus pattern")
    return msgs

def generate_feedback(history):
    dates = sorted(history.keys())
    if len(dates) < 2:
        return None
    diff = history[dates[-1]]["focus_minutes"] - history[dates[-2]]["focus_minutes"]
    return f"{'📈' if diff>0 else '📉'} {diff} min vs yesterday"

def generate_prevention_tips(level, insights):
    tips = []
    if level == "HIGH RISK (PLEASE USE THE PREVENTION MEASURES!!!)":
        tips += ["🛑 Take long break", "🚶 Step away"]
    elif level == "MEDIUM RISK (THERE IS STILL SOME TIME TO CHANGE.)":
        tips += ["⏳ Use Pomodoro", "📵 Reduce distractions"]
    else:
        tips += ["✅ Maintain routine"]
    return tips

# ---------------- UI ----------------
st.title("🌿 Focus Companion")
tab1, tab2 = st.tabs(["🎯 Live Focus", "📊 Insights"])

# ================= TAB 1 =================
with tab1:
    mode = st.radio("Mode", ["Normal Focus", "Pomodoro"], horizontal=True)
    st.session_state.mode = "pomodoro" if mode=="Pomodoro" else "normal"

    if st.button("▶ Start Session"):
        st.session_state.running = True
        st.session_state.session_started_at = time.time()
        st.session_state.last_time = time.time()
        st.session_state.pomodoro_start = time.time()

    if st.button("⛔ Stop & Save"):
        st.session_state.running = False
        if st.session_state.session_started_at is not None:
            st.session_state.focus_time = time.time() - st.session_state.session_started_at

        data = load_data()
        mins = int(st.session_state.focus_time)//60

        if mins>0:
            entry = data["history"].get(today, {
                "focus_minutes":0,
                "sessions":0,
                "avg_session":0,
                "hourly_focus":{}
            })

            entry["focus_minutes"] += mins
            entry["sessions"] += 1
            entry["avg_session"] = entry["focus_minutes"] / entry["sessions"]

            h = str(datetime.now().hour)
            entry["hourly_focus"][h] = entry["hourly_focus"].get(h,0) + mins

            data["history"][today] = entry
            data["streak"] = compute_streak(data["history"], today)
            data["garden"].append(get_plant(mins))

            save_data(data)

        st.session_state.focus_time = 0
        st.session_state.session_started_at = None
        st.rerun()

    if st.session_state.running:
        st.info("Allow browser camera access, then take a photo to check your presence.")
        camera_image = st.camera_input("Camera check", key="browser_camera")

        if st.session_state.session_started_at is not None:
            st.session_state.focus_time = time.time() - st.session_state.session_started_at
        minutes, seconds = divmod(int(st.session_state.focus_time), 60)
        st.metric("Session time", f"{minutes:02}:{seconds:02}")

        if camera_image is not None:
            image_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                if detect_face(frame):
                    st.success("Face detected — you are present.")
                else:
                    st.warning("No face detected. Try a brighter, front-facing photo.")

# ================= TAB 2 =================
with tab2:
    data = load_data()
    hist = data["history"]

    if not hist:
        st.info("No data yet")
    else:
        level,msg,insights = analyze_focus_patterns(hist)

        col1,col2,col3 = st.columns(3)
        with col1:
            st.metric("🔥 Streak", f"{data.get('streak',0)} days")
        with col2:
            latest = sorted(hist.keys())[-1]
            st.metric("⏱️ Today", f"{hist[latest]['focus_minutes']} min")
        with col3:
            fb = generate_feedback(hist)
            if fb: st.metric("📊 Progress", fb)

        st.divider()

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Status")
            st.write(level)
            st.write(insights)

        with col2:
            st.subheader("💡 Insights")
            for m in generate_human_insights(insights):
                st.write(m)

            st.subheader("🛡️ Prevention")
            for tip in generate_prevention_tips(level, insights):
                st.write(f"- {tip}")

        st.divider()

        st.subheader("📈 Trend")
        recent = sorted(hist.keys())[-7:]
        chart_data = {
            "Date":[d[5:] for d in recent],
            "Minutes":[hist[d]["focus_minutes"] for d in recent]
        }
        st.area_chart(data=chart_data, x="Date", y="Minutes")

        st.subheader("🌳 Garden")
        st.title(" ".join(data.get("garden", [])))
