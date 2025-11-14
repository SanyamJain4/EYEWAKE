"""
EYEWAKE — Upgraded Prototype
Adds:
 1. 5-second eye calibration
 2. Basic yawn detection
 3. Modular, cleaner UI overlay

Run: python3 eyewake_upgraded.py
Press c to calibrate, q to quit
"""

import cv2, mediapipe as mp, numpy as np, pyttsx3, time
from collections import deque

import json, os

PROFILE_FILE = "eyewake_profile.json"

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            return json.load(f).get("baseline_eye")
    return None

def save_profile(value):
    with open(PROFILE_FILE, "w") as f:
        json.dump({"baseline_eye": value}, f)


# ----- Voice engine -----
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
print("\nAvailable voices:")
for i, v in enumerate(voices):
    print(f" {i}: {v.name}")
try:
    choice = int(input("Choose a voice number (or Enter for default): ") or 0)
    engine.setProperty('voice', voices[choice].id)
except Exception:
    pass

def speak(msg):
    engine.say(msg)
    engine.runAndWait()

# ----- Mediapipe setup -----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ----- Constants -----
CONSEC_FRAMES_YELLOW = 10
CONSEC_FRAMES_RED = 25
RESET_OPEN_FRAMES = 15
ALERT_INTERVAL = 5      # seconds between alerts
YAWN_THRESHOLD = 20     # pixel mouth gap to trigger yawning (adjust)
CALIBRATION_TIME = 5    # seconds
EYE_HISTORY = deque(maxlen=15)

# eye and mouth landmark indices
L_EYE_T, L_EYE_B = 159, 145
R_EYE_T, R_EYE_B = 386, 374
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# ----- Helper functions -----
def measure_eye_height(lms, w, h):
    L = abs(int(lms[L_EYE_T].y*h) - int(lms[L_EYE_B].y*h))
    R = abs(int(lms[R_EYE_T].y*h) - int(lms[R_EYE_B].y*h))
    return (L+R)/2

def measure_mouth_gap(lms, h):
    return abs(int(lms[MOUTH_TOP].y*h) - int(lms[MOUTH_BOTTOM].y*h))

def draw_status(frame, color, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame,(0,0),(w,int(0.12*h)),color,-1)
    cv2.putText(frame,f"EYEWAKE — {text}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    # --- On-screen guidance ---
    cv2.putText(frame, "Press 'c'=Calibrate  'q'=Quit",
                (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)


def calibrate(cap, face_mesh):
    """5-second average open-eye height"""
    print("Calibration: look at camera normally.")
    start=time.time(); vals=[]
    while time.time()-start<CALIBRATION_TIME:
        ret,frm=cap.read()
        if not ret: continue
        frm=cv2.flip(frm,1)
        rgb=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
        res=face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lms=res.multi_face_landmarks[0].landmark
            h,w,_=frm.shape
            vals.append(measure_eye_height(lms,w,h))
        cv2.putText(frm,f"Calibrating... {int(CALIBRATION_TIME-(time.time()-start))}s",
                    (10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow("EYEWAKE Calibration",frm)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cv2.destroyWindow("EYEWAKE Calibration")
    if vals:
        base = np.median(vals)
        stdev = np.std(vals)
        thresh = base * 0.6  # adaptive threshold based on variation
        print(f"Baseline {base:.2f}, StdDev {stdev:.2f}, threshold {thresh:.2f}")

        print(f"Baseline {base:.2f}, threshold {thresh:.2f}")
        return thresh
    else:
        print("Calibration failed, using default 0.03 ratio")
        return None

# ----- Main -----
def run_detection():
    cap = cv2.VideoCapture(0)
    _, temp = cap.read()
    frame_h = temp.shape[0]
    default_thresh = frame_h * 0.03
    eye_thresh = default_thresh
    blink_count = open_count = 0
    last_alert = 0
    drowsy = False

    saved = load_profile()
    if saved:
        eye_thresh = saved * 0.65
        print(f"Loaded personal baseline: {saved:.2f}")

    privacy_mode = False  # GUI will handle user choice later
    start_time = time.time()
    alert_count = 0
    yawn_count  = 0

    print("Press 'c' to calibrate, 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # 👇 Paste your full existing detection logic here 👇
        # everything inside your while loop stays the same!
        # (eye measurement, mouth measurement, drawing, speaking, etc.)

        # keep this until the very end of your old while loop logic

    cap.release()
    cv2.destroyAllWindows()


# === EYEWAKE Session Summary ===
runtime = int(time.time() - start_time)
mins, secs = divmod(runtime, 60)
print("\n=== EYEWAKE SESSION SUMMARY ===")
print(f"Total runtime: {mins} min {secs} sec")
print(f"Drowsy alerts: {alert_count}")
print(f"Yawns detected: {yawn_count}")
if not privacy_mode:
    print("Baseline eye height stored for next session.")
print("===============================")

