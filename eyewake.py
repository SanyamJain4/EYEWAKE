# streamlit_app.py
"""
Streamlit front-end for EYEWAKE
- Uses Mediapipe face-mesh for landmarks
- Uses pyttsx3 for offline TTS (optional)
- Reads feature parameters and profile filename from your uploaded eyewake files
  (CALIBRATION_TIME, ALERT_INTERVAL, YAWN thresholds, counters, PROFILE_FILE)
References: eyewake.py / new.py constants.
"""
import streamlit as st
import threading
import time
import json
from collections import deque
import os

import cv2
import numpy as np
import mediapipe as mp

# Optional TTS (pyttsx3). If missing, app still runs but without voice.
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------------------------
# Parameters (taken from your uploaded eyewake.py / new.py)
# ---------------------------
PROFILE_FILE = "eyewake_profile.json"    # from eyewake.py
CONSEC_FRAMES_YELLOW = 10                # from eyewake.py / new.py
CONSEC_FRAMES_RED = 25
RESET_OPEN_FRAMES = 15
ALERT_INTERVAL = 5                       # seconds between spoken alerts
CALIBRATION_TIME = 5.0                   # seconds (5s calibration)
YAWN_PHYSICAL_PIXELS = 20                # pixel gap used in eyewake.py (2 cm * pixels_per_cm)
PIXELS_PER_CM = 10
YAWN_MIN_DURATION = 4.0
YAWN_COOLDOWN = 2.0
EYE_HISTORY_LEN = 15

# Streamlit-app defaults (kept but tuned to match the above)
CLOSED_FRAMES_TRIGGER = CONSEC_FRAMES_RED
EYE_COOLDOWN_SEC = ALERT_INTERVAL

# Mediapipe landmark indices (as used in your files)
L_EYE_T, L_EYE_B = 159, 145
R_EYE_T, R_EYE_B = 386, 374
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# ---------------------------
# Globals for background thread
# ---------------------------
frame_lock = threading.Lock()
latest_overlay = None
running_event = threading.Event()
stop_event = threading.Event()
calibrate_event = threading.Event()

eye_history = deque(maxlen=EYE_HISTORY_LEN)
mouth_history = deque(maxlen=EYE_HISTORY_LEN)

state_lock = threading.Lock()
state = {
    "status": "INIT",
    "yawns": 0,
    "alerts": 0,
    "eye_avg": 0.0,
    "mouth_avg": 0.0,
    "baseline": None,
    "eye_thresh": None,
    "calibrating": False,
    "calib_elapsed": 0.0
}

# TTS engine init (if available)
tts_engine = None
voices_list = []
if TTS_AVAILABLE:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    voices_list = tts_engine.getProperty('voices')
else:
    voices_list = []

# Mediapipe
mp_face = mp.solutions.face_mesh

# ---------------------------
# Helpers: profile load/save, speak
# ---------------------------
def load_profile():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                data = json.load(f)
                return data.get("baseline_eye")
        except Exception:
            return None
    return None

def save_profile(value):
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump({"baseline_eye": value}, f)
    except Exception:
        pass

def speak(text):
    if not TTS_AVAILABLE or tts_engine is None:
        return
    try:
        # run in non-blocking manner to avoid blocking UI thread
        def _s():
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception:
                pass
        t = threading.Thread(target=_s, daemon=True)
        t.start()
    except Exception:
        pass

def extract_point(lm, idx, w, h):
    p = lm[idx]
    return (int(p.x * w), int(p.y * h))

# ---------------------------
# Detection loop (background)
# ---------------------------
def detection_loop(cam_index=0, voice_index=None, save_profile_on_calib=True):
    global latest_overlay

    # set voice if provided
    if TTS_AVAILABLE and voice_index is not None:
        try:
            tts_engine.setProperty('voice', voices_list[voice_index].id)
        except Exception:
            pass

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # initialize from saved profile if available (match eyewake's behavior)
    baseline_saved = load_profile()
    if baseline_saved:
        with state_lock:
            state["baseline"] = baseline_saved
            # eyewake used saved * 0.65 as working threshold; emulate that
            state["eye_thresh"] = baseline_saved * 0.65

    closed_frames = 0
    yawning = False
    yawn_start = 0.0
    last_yawn_time = 0.0
    last_alert_time = 0.0

    running_event.set()
    stop_event.clear()
    calib_samples = []
    calib_start_time = None

    while running_event.is_set() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        eye_val = 0.0
        mouth_val = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                lt = extract_point(lm, L_EYE_T, w, h); lb = extract_point(lm, L_EYE_B, w, h)
                rt = extract_point(lm, R_EYE_T, w, h); rb = extract_point(lm, R_EYE_B, w, h)
                left = abs(lt[1] - lb[1]) / max(1.0, h)
                right = abs(rt[1] - rb[1]) / max(1.0, h)
                eye_val = (left + right) / 2.0
            except Exception:
                eye_val = 0.0

            try:
                mt = extract_point(lm, MOUTH_TOP, w, h)
                mb = extract_point(lm, MOUTH_BOTTOM, w, h)
                mouth_val = abs(mb[1] - mt[1]) / max(1.0, h)
            except Exception:
                mouth_val = 0.0

            # draw small markers
            for idx in (L_EYE_T, L_EYE_B, R_EYE_T, R_EYE_B, MOUTH_TOP, MOUTH_BOTTOM):
                p = extract_point(lm, idx, w, h)
                cv2.circle(frame, p, 2, (0,255,0), -1)

        # smoothing
        eye_history.append(eye_val)
        mouth_history.append(mouth_val)
        eye_avg = float(np.mean(eye_history)) if len(eye_history) > 0 else 0.0
        mouth_avg = float(np.mean(mouth_history)) if len(mouth_history) > 0 else 0.0

        # handle calibration event (collect open-eye samples for CALIBRATION_TIME seconds)
        if calibrate_event.is_set():
            if calib_start_time is None:
                calib_start_time = time.time()
                calib_samples = []
                with state_lock:
                    state["calibrating"] = True
                    state["calib_elapsed"] = 0.0
                speak("Calibration started")
            else:
                with state_lock:
                    state["calib_elapsed"] = time.time() - calib_start_time
            calib_samples.append(eye_avg)
            if (time.time() - calib_start_time) >= CALIBRATION_TIME:
                # finish calibration (emulate eyewake: use median and threshold base * 0.6)
                baseline = float(np.median(calib_samples)) if len(calib_samples)>0 else eye_avg
                thresh = baseline * 0.6
                with state_lock:
                    state["baseline"] = baseline
                    state["eye_thresh"] = thresh
                    state["calibrating"] = False
                    state["calib_elapsed"] = 0.0
                # optionally save profile
                if save_profile_on_calib:
                    save_profile(baseline)
                calibrate_event.clear()
                calib_start_time = None
                calib_samples = []
                speak("Calibration complete")
        else:
            with state_lock:
                state["calib_elapsed"] = 0.0
                state["calibrating"] = False

        # Yawn detection using physical pixel threshold from eyewake (2 cm * pixels per cm)
        # convert mouth pixel gap using frame height -> approximate pixels: YAWN_PHYSICAL_PIXELS (from eyewake)
        # Note: eyewake used pixel gap measured directly; here we compare normalized mouth_avg to approximate normalized threshold.
        approx_pixel_threshold = (YAWN_PHYSICAL_PIXELS / float(max(1, h)))  # normalized threshold
        if mouth_avg > approx_pixel_threshold:
            if not yawning:
                yawning = True
                yawn_start = time.time()
            else:
                if time.time() - yawn_start >= YAWN_MIN_DURATION:
                    if time.time() - last_yawn_time >= YAWN_COOLDOWN:
                        with state_lock:
                            state["yawns"] += 1
                        last_yawn_time = time.time()
                        yawning = False
                        speak("Yawn detected")
        else:
            yawning = False

        # Eye-closure / drowsiness detection, use eye_thresh if set otherwise fallback to small default
        with state_lock:
            eye_thresh = state.get("eye_thresh", None)
        if eye_thresh is None:
            # fallback: use frame-height relative default similar to eyewake default_thresh = frame_h * 0.03
            # normalized default threshold (0.03 * frame_h / frame_h = 0.03)
            eye_thresh_norm = 0.03
        else:
            # state stores absolute pixel baseline originally; but in our normalized flow baseline is normalized already.
            # eyewake stored baseline in pixels; here baseline / h would have been normalized. We saved baseline in normalized units.
            # We assume saved baseline was measured as pixel value divided by height – consistent with how we compute.
            eye_thresh_norm = eye_thresh

        # detect closed frames (normalized)
        if 0 < eye_avg < eye_thresh_norm:
            closed_frames += 1
        else:
            closed_frames = 0

        now = time.time()
        if closed_frames >= CLOSED_FRAMES_TRIGGER:
            if now - last_alert_time > EYE_COOLDOWN_SEC:
                with state_lock:
                    state["alerts"] += 1
                    state["status"] = "DROWSY"
                last_alert_time = now
                speak("You look drowsy. Please take a break.")
        else:
            # not drowsy; if currently calibrating or yawning keep that status priority
            with state_lock:
                if state.get("calibrating", False):
                    state["status"] = "CALIBRATING"
                elif mouth_avg > approx_pixel_threshold:
                    state["status"] = "YAWNING"
                else:
                    state["status"] = "ACTIVE"

        with state_lock:
            state["eye_avg"] = eye_avg
            state["mouth_avg"] = mouth_avg

        # overlays
        overlay = frame.copy()
        cv2.putText(overlay, f"Status: {state['status']}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(overlay, f"Eye: {eye_avg:.3f}  Thresh: {state.get('eye_thresh',0):.3f}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(overlay, f"Mouth: {mouth_avg:.3f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(overlay, f"Yawns: {state['yawns']}  Alerts: {state['alerts']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        with frame_lock:
            latest_overlay = overlay.copy()

        time.sleep(0.02)

    # cleanup
    cap.release()
    face_mesh.close()
    running_event.clear()
    with state_lock:
        state["status"] = "STOPPED"
    speak("Detection stopped")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Eyewake (Streamlit)", layout="centered")
st.title("🕶️ Eyewake — Streamlit (TTS + saved profile)")

# Sidebar: voice selection & settings
with st.sidebar:
    st.header("Settings")
    tts_enabled = st.checkbox("Enable voice alerts (pyttsx3)", value=TTS_AVAILABLE and True)
    if tts_enabled and TTS_AVAILABLE:
        # build voice options
        voice_names = [f"{i}: {v.name}" for i, v in enumerate(voices_list)]
        voice_sel = st.selectbox("Choose voice", options=voice_names, index=0)
        selected_voice_index = int(voice_sel.split(":")[0])
    else:
        selected_voice_index = None

    st.markdown("---")
    st.write("Feature parameters (loaded from eyewake.py):")
    st.write(f"- Calibration time: **{CALIBRATION_TIME}s**")
    st.write(f"- Alert interval: **{ALERT_INTERVAL}s**")
    st.write(f"- Yawn pixel threshold: **{YAWN_PHYSICAL_PIXELS}px** (approx)")
    use_saved_profile = st.checkbox("Auto-save calibration to profile", value=True)
    st.markdown("---")
    if st.button("Reset saved profile"):
        if os.path.exists(PROFILE_FILE):
            os.remove(PROFILE_FILE)
            st.success("Profile removed.")
        else:
            st.info("No profile to remove.")

# Main controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Start") and not running_event.is_set():
        # start detection thread
        stop_event.clear()
        running_event.set()
        th = threading.Thread(target=detection_loop, args=(0, selected_voice_index, use_saved_profile), daemon=True)
        th.start()
        speak("Detection started") if tts_enabled and TTS_AVAILABLE else None

with col2:
    if st.button("Calibrate"):
        if running_event.is_set():
            calibrate_event.set()
        else:
            st.warning("Start detection first")

with col3:
    if st.button("Stop"):
        stop_event.set()
        running_event.clear()

# Display area
img_slot = st.empty()
status_slot = st.empty()
meta_slot = st.empty()

# If a profile exists, show it
saved_profile = load_profile()
if saved_profile:
    st.info(f"Loaded saved baseline (from {PROFILE_FILE}): {saved_profile:.4f} (used as baseline)")

# Main UI loop: update frames and status
try:
    while True:
        if not running_event.is_set():
            status_slot.info("Status: NOT RUNNING")
            img_slot.image(np.zeros((480,640,3), dtype=np.uint8), channels="BGR", use_container_width=True)
            break

        with frame_lock:
            frame_show = None if latest_overlay is None else latest_overlay.copy()
        if frame_show is not None:
            img_slot.image(cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB), use_container_width=True)

        with state_lock:
            s = dict(state)  # shallow copy

        status_slot.markdown(f"**Status:** {s['status']}  •  **Yawns:** {s['yawns']}  •  **Alerts:** {s['alerts']}")
        if s.get("calibrating", False):
            meta_slot.info(f"Calibrating... {s['calib_elapsed']:.1f}/{CALIBRATION_TIME}s")
        else:
            meta = f"Eye avg: {s['eye_avg']:.3f}  |  Mouth avg: {s['mouth_avg']:.3f}"
            if s.get("baseline") is not None:
                meta += f"  |  Baseline: {s['baseline']:.4f}  |  Threshold: {s.get('eye_thresh'):.4f}"
            meta_slot.write(meta)

        time.sleep(0.12)

except Exception as e:
    stop_event.set()
    running_event.clear()
    st.error(f"Stopped due to: {e}")
    raise
