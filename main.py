import cv2
import time
import numpy as np
import json
import os
from datetime import date, timedelta
import matplotlib.pyplot as plt

# -------------------- DATA --------------------
DATA_FILE = "data.json"

def load_data():
    default = {
        "streak": 0,
        "last_date": str(date.today()),
        "garden": [],
        "history": {}
    }

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            for k in default:
                if k not in data:
                    data[k] = default[k]
            return data
        except:
            print("⚠️ Resetting data")

    return default

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

data = load_data()

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# -------------------- TIME --------------------
total_focus_time = 0
last_frame_time = time.time()

prev_frame = None
last_motion_time = time.time()
last_seen_time = time.time()
last_active_time = time.time()

# -------------------- PARAMETERS --------------------
MOTION_THRESHOLD = 4000
PRESENCE_THRESHOLD = 2000

IDLE_TIME = 8
FOCUS_BUFFER = 8
PRESENCE_BUFFER = 6

today = date.today()

# -------------------- TRACKING --------------------
break_count = 0
last_state = "Idle"
distraction_score = 0

session_lengths = []
current_session_start = None

# -------------------- HELPERS --------------------
def avg_session(sessions):
    return (sum(sessions)/len(sessions))/60 if sessions else 0

def last_n_days(history, n=3):
    values = []
    for i in range(n):
        d = str(date.today() - timedelta(days=i))
        if d in history:
            values.append(history[d]["focus_minutes"])
    return values

def trend_analysis(history, today_minutes):
    last3 = last_n_days(history, 3)
    if not last3:
        return "No trend yet", 0

    avg3 = sum(last3) / len(last3)

    if today_minutes > avg3:
        return "Improving trend 📈", 1
    else:
        return "Declining trend 📉", -1

def explain_change(history, minutes, breaks):
    yesterday = str(date.today() - timedelta(days=1))
    if yesterday in history:
        prev = history[yesterday]
        reason = []
        if minutes > prev["focus_minutes"]:
            reason.append("more focus time")
        if breaks < prev["breaks"]:
            reason.append("fewer breaks")
        if not reason:
            return "No major change"
        return "Improved due to " + ", ".join(reason)
    return "No previous comparison"

def classify_user(sessions, breaks):
    avg = avg_session(sessions)
    if avg > 25:
        return "Deep Worker"
    if breaks > 5:
        return "Frequent Switcher"
    return "Consistent Performer"

def burnout_level(minutes, breaks, distraction, sessions, trend_flag):
    avg = avg_session(sessions)
    score = 0

    if minutes > 120: score += 2
    if breaks > 8: score += 2
    if distraction > 15: score += 2
    if avg < 8: score += 2
    if trend_flag == -1: score += 1   # NEW: trend-aware

    if score <= 2: return "Low"
    elif score <= 5: return "Moderate"
    else: return "High"

def generate_graph(history):
    dates = list(history.keys())[-7:]
    focus = [history[d]["focus_minutes"] for d in dates]

    if not dates:
        return

    plt.figure()
    plt.plot(dates, focus)
    plt.title("Focus Trend")
    plt.xlabel("Date")
    plt.ylabel("Minutes")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("focus_trend.png")
    plt.close()

# -------------------- LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    delta = current_time - last_frame_time
    last_frame_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    motion = np.sum(thresh)
    area = motion / 255

    if motion > MOTION_THRESHOLD:
        last_motion_time = current_time

    if area > PRESENCE_THRESHOLD:
        last_seen_time = current_time

    user_present = (current_time - last_seen_time) < PRESENCE_BUFFER
    motion_active = (current_time - last_motion_time) < IDLE_TIME

    if user_present and motion_active:
        last_active_time = current_time

    active = user_present and ((current_time - last_active_time) < FOCUS_BUFFER)

    if active:
        total_focus_time += delta

    minutes = int(total_focus_time) // 60
    seconds = int(total_focus_time) % 60

    status = "Focused" if active else "Idle"

    if status != last_state:
        if last_state == "Focused":
            break_count += 1
            distraction_score += 1
        last_state = status

    if motion > 20000:
        distraction_score += 1

    # ---------------- ANALYTICS ----------------
    history = data.get("history", {})

    trend_text, trend_flag = trend_analysis(history, minutes)
    explanation = explain_change(history, minutes, break_count)
    user_type = classify_user(session_lengths, break_count)
    burnout = burnout_level(minutes, break_count, distraction_score, session_lengths, trend_flag)

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, f"Focus: {minutes}m {seconds}s", (20,40), 0, 0.7, (0,255,0),2)
    cv2.putText(frame, f"Trend: {trend_text}", (20,80), 0, 0.6, (255,255,0),2)
    cv2.putText(frame, f"Why: {explanation}", (20,110), 0, 0.5, (200,255,200),2)
    cv2.putText(frame, f"Burnout: {burnout}", (20,140), 0, 0.6, (0,150,255),2)
    cv2.putText(frame, f"Type: {user_type}", (20,170), 0, 0.6, (255,150,150),2)

    cv2.imshow("Focus Companion", frame)
    prev_frame = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

    try:
        if cv2.getWindowProperty("Focus Companion", cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

# ---------------- SAVE ----------------
today_str = str(today)

if "history" not in data:
    data["history"] = {}

data["history"][today_str] = {
    "focus_minutes": minutes,
    "breaks": break_count,
    "distraction": distraction_score
}

generate_graph(data["history"])  # NEW

save_data(data)

cap.release()
cv2.destroyAllWindows()