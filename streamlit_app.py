# ============================================================
#  EYEWAKE — Streamlit FINAL VERSION
#  Pixel-based eye + yawn detection
#  Hybrid audio (pyttsx3 → gTTS → audio in browser)
#  NO storage, NO profile file
#  Fully crash-proof (all None safe)
# ============================================================

import streamlit as st
import threading
import time
from collections import deque
import queue
import tempfile
import uuid
import traceback
import os

import cv2
import numpy as np
import mediapipe as mp

# ------------------------------------------------------------
# AUDIO SYSTEM (pyttsx3 → gTTS fallback → browser audio)
# ------------------------------------------------------------
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

audio_queue = queue.Queue()

def synthesize_audio_bytes(text):
    """Try pyttsx3 → gTTS → return audio bytes."""
    uid = uuid.uuid4().hex

    # --- 1. pyttsx3 offline ---
    if PYTTSX3_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            out_file = os.path.join(tempfile.gettempdir(), f"tts_{uid}.wav")
            engine.save_to_file(text, out_file)
            engine.runAndWait()
            data = open(out_file, "rb").read()
            os.remove(out_file)
            return data, "audio/wav"
        except:
            pass

    # --- 2. gTTS fallback ---
    if GTTS_AVAILABLE:
        try:
            out_file = os.path.join(tempfile.gettempdir(), f"tts_{uid}.mp3")
            gTTS(text=text, lang="en").save(out_file)
            data = open(out_file, "rb").read()
            os.remove(out_file)
            return data, "audio/mp3"
        except:
            pass

    return None, None


def queue_audio(text):
    audio_queue.put(text)


# ------------------------------------------------------------
# CONSTANTS (Pixel-based detection)
# ------------------------------------------------------------
CONSEC_FRAMES_RED = 25
ALERT_INTERVAL = 5.0
CALIBRATION_TIME = 5.0

YAWN_PHYS_PIXELS = 20
YAWN_MIN_DURATION = 4.0
YAWN_COOLDOWN = 2.0

EYE_HISTORY_LEN = 15
DEFAULT_EYE_THRESH = 12    # px before calibration

# Mediapipe landmark indices
L_EYE_T, L_EYE_B = 159, 145
R_EYE_T, R_EYE_B = 386, 374
MOUTH_TOP, MOUTH_BOTTOM = 13, 14


# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------
frame_lock = threading.Lock()
latest_overlay = None

running_event = threading.Event()
stop_event = threading.Event()
calibrate_event = threading.Event()

eye_history = deque(maxlen=EYE_HISTORY_LEN)
mouth_history = deque(maxlen=EYE_HISTORY_LEN)

state = {
    "status": "INIT",
    "yawns": 0,
    "alerts": 0,
    "eye_avg": 0.0,
    "mouth_avg": 0.0,
    "baseline": None,
    "eye_thresh": None,
    "calibrating": False,
    "calib_elapsed": 0.0,
}


# ------------------------------------------------------------
# Mediapipe Helper
# ------------------------------------------------------------
mp_face = mp.solutions.face_mesh

def extract_point(lm, idx, w, h):
    p = lm[idx]
    return (int(p.x * w), int(p.y * h))


# ============================================================
# DETECTION LOOP — THREAD
# ============================================================
def detection_loop(cam_index=0):

    global latest_overlay

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, test = cap.read()
    if not ret:
        state["status"] = "CAMERA ERROR"
        queue_audio("Camera not detected")
        running_event.clear()
        return

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    closed_frames = 0
    yawning = False
    yawn_start = 0
    last_yawn_time = 0
    last_alert = 0

    calib_start = None
    calib_samples = []

    running_event.set()
    stop_event.clear()

    while running_event.is_set() and not stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # default values (avoid None)
        eye_pixels = 0.0
        mouth_pixels = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Pixel eye height
            lt = extract_point(lm, L_EYE_T, w, h)
            lb = extract_point(lm, L_EYE_B, w, h)
            rt = extract_point(lm, R_EYE_T, w, h)
            rb = extract_point(lm, R_EYE_B, w, h)

            left_h = abs(lt[1] - lb[1])
            right_h = abs(rt[1] - rb[1])
            eye_pixels = (left_h + right_h) / 2.0

            # Pixel mouth gap
            mt = extract_point(lm, MOUTH_TOP, w, h)
            mb = extract_point(lm, MOUTH_BOTTOM, w, h)
            mouth_pixels = abs(mb[1] - mt[1])

            # markers
            for idx in (L_EYE_T, L_EYE_B, R_EYE_T, R_EYE_B, MOUTH_TOP, MOUTH_BOTTOM):
                p = extract_point(lm, idx, w, h)
                cv2.circle(frame, p, 2, (0,255,0), -1)

        # smoothing
        eye_history.append(eye_pixels)
        mouth_history.append(mouth_pixels)

        eye_avg = float(np.mean(eye_history)) if eye_history else 0.0
        mouth_avg = float(np.mean(mouth_history)) if mouth_history else 0.0

        # ---------------- CALIBRATION ----------------
        if calibrate_event.is_set():
            if calib_start is None:
                calib_start = time.time()
                calib_samples = []
                state["calibrating"] = True
                queue_audio("Calibration started")
            else:
                state["calib_elapsed"] = time.time() - calib_start

            calib_samples.append(eye_pixels)

            if time.time() - calib_start >= CALIBRATION_TIME:
                baseline = float(np.median(calib_samples)) if calib_samples else 12.0
                thresh = baseline * 0.6

                state["baseline"] = baseline
                state["eye_thresh"] = thresh
                state["calibrating"] = False
                state["calib_elapsed"] = 0.0

                calibrate_event.clear()
                calib_start = None
                queue_audio("Calibration complete")

        # ---------------- YAWN DETECTION ----------------
        if mouth_avg > YAWN_PHYS_PIXELS:
            if not yawning:
                yawning = True
                yawn_start = time.time()
            else:
                if time.time() - yawn_start >= YAWN_MIN_DURATION:
                    if time.time() - last_yawn_time >= YAWN_COOLDOWN:
                        state["yawns"] += 1
                        last_yawn_time = time.time()
                        yawning = False
                        queue_audio("Yawn detected")
        else:
            yawning = False

        # ---------------- EYE CLOSURE DETECTION ----------------
        thresh = state.get("eye_thresh") or DEFAULT_EYE_THRESH

        if eye_avg < thresh:
            closed_frames += 1
        else:
            closed_frames = 0

        now = time.time()

        if closed_frames >= CONSEC_FRAMES_RED:
            if now - last_alert >= ALERT_INTERVAL:
                state["alerts"] += 1
                state["status"] = "DROWSY"
                last_alert = now
                queue_audio("You look drowsy. Please take a break.")
        else:
            if state["calibrating"]:
                state["status"] = "CALIBRATING"
            elif mouth_avg > YAWN_PHYS_PIXELS:
                state["status"] = "YAWNING"
            else:
                state["status"] = "ACTIVE"

        # update state values
        state["eye_avg"] = eye_avg
        state["mouth_avg"] = mouth_avg

        # ---------------- OVERLAY ----------------
        overlay = frame.copy()

        cv2.putText(overlay, f"Status: {state['status']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(overlay,
                    f"Eye(px): {eye_avg:.1f}  Thresh(px): {thresh:.1f}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        cv2.putText(overlay,
                    f"Mouth(px): {mouth_avg:.1f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        cv2.putText(overlay,
                    f"Yawns: {state['yawns']}  Alerts: {state['alerts']}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        with frame_lock:
            latest_overlay = overlay.copy()

        time.sleep(0.02)

    # cleanup
    cap.release()
    state["status"] = "STOPPED"
    queue_audio("Detection stopped")


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Eyewake", layout="centered")
st.title("🕶️ EYEWAKE — Pixel-based Detection (Final Version)")

with st.sidebar:
    st.header("Camera Settings")
    cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start") and not running_event.is_set():
        stop_event.clear()
        running_event.set()
        threading.Thread(target=detection_loop, args=(cam_idx,), daemon=True).start()
        queue_audio("Detection started")

with col2:
    if st.button("Calibrate"):
        if running_event.is_set():
            calibrate_event.set()
        else:
            st.warning("Start detection first.")

with col3:
    if st.button("Stop"):
        running_event.clear()
        stop_event.set()
        queue_audio("Stopping")

img_slot = st.empty()
status_slot = st.empty()
meta_slot = st.empty()


# ============================================================
# MAIN LOOP (UI + AUDIO)
# ============================================================

try:
    while True:

        # -------- Audio playback --------
        while not audio_queue.empty():
            msg = audio_queue.get()
            audio_bytes, mime = synthesize_audio_bytes(msg)
            if audio_bytes:
                st.audio(audio_bytes, format=mime)
            audio_queue.task_done()

        # -------- Video feed --------
        if not running_event.is_set():
            img_slot.image(
                np.zeros((480,640,3), dtype=np.uint8),
                channels="BGR",
                use_container_width=True
            )
            status_slot.info("Status: Not running")
            break

        # frame
        with frame_lock:
            frame_show = None if latest_overlay is None else latest_overlay.copy()

        if frame_show is not None:
            img_slot.image(
                cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        # safe values
        s = dict(state)

        status_slot.markdown(
            f"**Status:** {s.get('status','-')} • "
            f"**Yawns:** {s.get('yawns',0)} • "
            f"**Alerts:** {s.get('alerts',0)}"
        )

        # safe numeric formatting
        eye_val = s.get("eye_avg") or 0.0
        mouth_val = s.get("mouth_avg") or 0.0
        baseline = s.get("baseline") or 0.0
        thr = s.get("eye_thresh") or DEFAULT_EYE_THRESH

        if s.get("calibrating"):
            meta_slot.info(
                f"Calibrating… {s.get('calib_elapsed',0):.1f}/{CALIBRATION_TIME}s"
            )
        else:
            meta_slot.write(
                f"Eye(px): {eye_val:.1f} | "
                f"Mouth(px): {mouth_val:.1f} | "
                f"Baseline: {baseline:.1f} | "
                f"Threshold(px): {thr:.1f}"
            )

        time.sleep(0.12)

except Exception as e:
    st.error(f"Error: {e}")
    traceback.print_exc()
