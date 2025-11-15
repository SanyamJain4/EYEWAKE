# ============================================================
#  EYEWAKE — Streamlit FINAL VERSION
#  Pixel-based eye + yawn detection
#  With mediapipe import safety (for Streamlit Cloud)
# ============================================================

import streamlit as st
import time
import threading
import queue
import uuid
import tempfile
import cv2
import numpy as np

# ------------------------------------------------------------
# TRY IMPORT MEDIAPIPE SAFELY
# ------------------------------------------------------------

MEDIAPIPE_AVAILABLE = True
try:
    import mediapipe as mp
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    MP_ERROR = str(e)

# ------------------------------------------------------------
# AUDIO MODULES
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

def make_audio(text):
    """Synthesize audio fallback system."""
    uid = uuid.uuid4().hex
    # pyttsx3
    if PYTTSX3_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            out_file = tempfile.gettempdir() + f"/tts_{uid}.wav"
            engine.save_to_file(text, out_file)
            engine.runAndWait()
            data = open(out_file, "rb").read()
            return data, "audio/wav"
        except:
            pass

    # gTTS fallback
    if GTTS_AVAILABLE:
        try:
            out_file = tempfile.gettempdir() + f"/tts_{uid}.mp3"
            gTTS(text).save(out_file)
            data = open(out_file, "rb").read()
            return data, "audio/mp3"
        except:
            pass

    return None, None


def queue_audio(text):
    audio_queue.put(text)


# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------
running_event = threading.Event()
stop_event = threading.Event()
frame_lock = threading.Lock()

latest_overlay = None
state = {
    "status": "NOT RUNNING",
    "yawns": 0,
    "alerts": 0,
    "eye_avg": 0.0,
    "mouth_avg": 0.0,
}


# ------------------------------------------------------------
# DETECTION LOOP
# ------------------------------------------------------------
def detection_loop(index):

    global latest_overlay
    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        state["status"] = "CAMERA ERROR"
        queue_audio("Camera not detected")
        running_event.clear()
        return

    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    )

    while running_event.is_set() and not stop_event.is_set():

        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        eye_height = 0
        mouth_gap = 0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # Eye + mouth points
            def p(i):
                return int(lm[i].x * w), int(lm[i].y * h)

            lt = p(159)
            lb = p(145)
            rt = p(386)
            rb = p(374)

            eye_height = (abs(lt[1]-lb[1]) + abs(rt[1]-rb[1])) / 2

            mt = p(13)
            mb = p(14)
            mouth_gap = abs(mb[1] - mt[1])

            # draw
            for i in [159,145,386,374,13,14]:
                cv2.circle(frame, p(i), 2, (0,255,0), -1)

        state["eye_avg"] = eye_height
        state["mouth_avg"] = mouth_gap

        if eye_height < 10:
            state["alerts"] += 1
            queue_audio("Eyes closing detected")

        if mouth_gap > 25:
            state["yawns"] += 1
            queue_audio("Yawn detected")

        state["status"] = "RUNNING"

        overlay = frame.copy()
        cv2.putText(overlay, f"Eye(px): {eye_height:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        with frame_lock:
            latest_overlay = overlay.copy()

        time.sleep(0.02)

    cap.release()
    state["status"] = "STOPPED"


# ============================================================
# UI SECTION
# ============================================================

st.set_page_config(page_title="Eyewake", layout="centered")
st.title("🕶️ Eyewake — Drowsiness Detector")

# ------------------ CHECK MEDIAPIPE FIRST -------------------
if not MEDIAPIPE_AVAILABLE:
    st.error("❌ MediaPipe failed to import. This app requires MediaPipe!")
    st.code(MP_ERROR)
    st.info("Fix: Ensure Python = 3.11 and mediapipe==0.10.14 in requirements.txt")
    st.stop()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
cam_idx = st.sidebar.number_input("Camera Index", 0, 5, 0)

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Detection") and not running_event.is_set():
        running_event.set()
        stop_event.clear()
        threading.Thread(target=detection_loop, args=(cam_idx,), daemon=True).start()
        queue_audio("Detection started")

with col2:
    if st.button("Stop"):
        running_event.clear()
        stop_event.set()
        queue_audio("Stopping")


img_slot = st.empty()
info_slot = st.empty()

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    # Audio
    while not audio_queue.empty():
        msg = audio_queue.get()
        data, fmt = make_audio(msg)
        if data:
            st.audio(data, format=fmt)
        audio_queue.task_done()

    # Video
    if running_event.is_set():
        with frame_lock:
            frame = None if latest_overlay is None else latest_overlay.copy()

        if frame is not None:
            img_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        s = state.copy()
        info_slot.info(
            f"Status: {s['status']} | Eye(px): {s['eye_avg']:.1f} | "
            f"Mouth(px): {s['mouth_avg']:.1f} | "
            f"Yawns: {s['yawns']} | Alerts: {s['alerts']}"
        )
    else:
        img_slot.image(np.zeros((480,640,3), dtype=np.uint8))
        info_slot.info("Status: Not running")
        break

    time.sleep(0.12)
