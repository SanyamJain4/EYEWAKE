# streamlit_app.py
"""
EYEWAKE — Camera-only Streamlit app (no cv2)
- Uses st.camera_input() for browser camera
- Uses MediaPipe FaceMesh for landmarks
- Pixel-based eye + yawn detection, calibration, TTS queue (pyttsx3 -> gTTS)
- Uses PIL for overlays (no cv2 import)
"""

import streamlit as st
import time
import queue
import uuid
import os
import tempfile
import traceback
from collections import deque
from io import BytesIO

# imaging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# mediapipe (required)
try:
    import mediapipe as mp
except Exception as e:
    mp = None
    MP_IMPORT_ERROR = traceback.format_exc()

# Optional offline TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

# gTTS fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ----------------------------
# CONSTANTS (same as your original)
CONSEC_FRAMES_RED = 25
ALERT_INTERVAL = 5.0
CALIBRATION_TIME = 5.0

YAWN_PHYS_PIXELS = 20
YAWN_MIN_DURATION = 4.0
YAWN_COOLDOWN = 2.0

EYE_HISTORY_LEN = 15
DEFAULT_EYE_THRESH = 12.0    # px before calibration

# Mediapipe landmark indices (same)
L_EYE_T, L_EYE_B = 159, 145
R_EYE_T, R_EYE_B = 386, 374
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# ----------------------------
# Shared app state via session_state
if "eyewake_state" not in st.session_state:
    st.session_state.eyewake_state = {
        "status": "INIT",
        "yawns": 0,
        "alerts": 0,
        "eye_avg": 0.0,
        "mouth_avg": 0.0,
        "baseline": None,
        "eye_thresh": None,
        "calibrating": False,
        "calib_elapsed": 0.0,
        "eye_history": deque(maxlen=EYE_HISTORY_LEN),
        "mouth_history": deque(maxlen=EYE_HISTORY_LEN),
        # local runtime flags (not persisted)
        "_yawning": False,
        "_yawn_start": 0.0,
        "_last_yawn_time": 0.0,
        "_closed_frames": 0,
        "_last_alert": 0.0,
        "_calib_start": None,
        "_calib_samples": [],
    }

state = st.session_state.eyewake_state

# audio queue for main thread to play
audio_queue = queue.Queue()

def synthesize_audio_bytes(text):
    """Try pyttsx3 → gTTS → return audio bytes and mime or (None, None)."""
    uid = uuid.uuid4().hex

    # 1. pyttsx3 offline
    if PYTTSX3_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            out_file = os.path.join(tempfile.gettempdir(), f"tts_{uid}.wav")
            engine.save_to_file(text, out_file)
            engine.runAndWait()
            with open(out_file, "rb") as f:
                data = f.read()
            os.remove(out_file)
            return data, "audio/wav"
        except Exception:
            pass

    # 2. gTTS fallback
    if GTTS_AVAILABLE:
        try:
            out_file = os.path.join(tempfile.gettempdir(), f"tts_{uid}.mp3")
            gTTS(text=text, lang="en").save(out_file)
            with open(out_file, "rb") as f:
                data = f.read()
            os.remove(out_file)
            return data, "audio/mp3"
        except Exception:
            pass

    return None, None

def queue_audio(text):
    audio_queue.put(text)

# ----------------------------
# Helpers: mediapipe landmark extraction and drawing (PIL)
def lm_to_pixel(lm, idx, w, h):
    p = lm[idx]
    return int(p.x * w), int(p.y * h)

def draw_overlay_pil(image_pil, landmarks, w, h, state_snapshot):
    """Draw small markers and text overlay on PIL image and return PIL image."""
    draw = ImageDraw.Draw(image_pil)
    # try to load a default font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # draw landmarks markers
    if landmarks:
        lm = landmarks
        for idx in (L_EYE_T, L_EYE_B, R_EYE_T, R_EYE_B, MOUTH_TOP, MOUTH_BOTTOM):
            x, y = lm_to_pixel(lm, idx, w, h)
            r = 3
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,255,0))

    # status text
    text_lines = [
        f"Status: {state_snapshot.get('status','-')}",
        f"Eye(px): {state_snapshot.get('eye_avg',0.0):.1f}  Thresh(px): {state_snapshot.get('eye_thresh') or DEFAULT_EYE_THRESH:.1f}",
        f"Mouth(px): {state_snapshot.get('mouth_avg',0.0):.1f}",
        f"Yawns: {state_snapshot.get('yawns',0)}  Alerts: {state_snapshot.get('alerts',0)}"
    ]
    y0 = 6
    for line in text_lines:
        draw.text((6, y0), line, fill=(255,255,0), font=font)
        y0 += 18

    return image_pil

# ----------------------------
# UI
st.set_page_config(page_title="Eyewake (camera-only)", layout="centered")
st.title("🕶️ EYEWAKE — Camera-only (no cv2)")

# Mediapipe availability check
if mp is None:
    st.error("MediaPipe failed to import. The app requires `mediapipe` installed.")
    with st.expander("MediaPipe import traceback"):
        st.code(MP_IMPORT_ERROR)
    st.stop()

mp_face = mp.solutions.face_mesh

with st.sidebar:
    st.header("Controls")
    st.write("Use your browser camera to take a snapshot and analyze.")
    # simple option to auto-calibrate on next N captures, but we keep the button for manual calibrate
    cam_note = st.info("Camera works only inside the browser. Allow camera permission when prompted.")

col1, col2, col3 = st.columns(3)

start_btn = col1.button("Take Snapshot")
calib_btn = col2.button("Calibrate (from this shot)")
reset_btn = col3.button("Reset Stats")

# Display slots
img_slot = st.empty()
status_slot = st.empty()
meta_slot = st.empty()

# Reset logic
if reset_btn:
    # reset persistent values
    state.update({
        "status": "INIT",
        "yawns": 0,
        "alerts": 0,
        "eye_avg": 0.0,
        "mouth_avg": 0.0,
        "baseline": None,
        "eye_thresh": None,
        "calibrating": False,
        "calib_elapsed": 0.0,
        "eye_history": deque(maxlen=EYE_HISTORY_LEN),
        "mouth_history": deque(maxlen=EYE_HISTORY_LEN),
        "_yawning": False,
        "_yawn_start": 0.0,
        "_last_yawn_time": 0.0,
        "_closed_frames": 0,
        "_last_alert": 0.0,
        "_calib_start": None,
        "_calib_samples": [],
    })
    st.success("Stats reset")

# Camera input (single snapshot)
img_file = st.camera_input("Click to take a picture", key="camera_input")

def process_frame_pil(pil_image):
    """
    Input: PIL image in RGB mode.
    Returns: annotated PIL image, a state snapshot dict update
    """
    w, h = pil_image.size
    # convert to numpy RGB
    rgb = np.array(pil_image)
    # process via mediapipe (expects RGB uint8)
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5)
    results = face_mesh.process(rgb)
    face_mesh.close()

    eye_pixels = 0.0
    mouth_pixels = 0.0
    landmarks = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # compute eye pixel heights
        lt = lm_to_pixel(landmarks, L_EYE_T, w, h)
        lb = lm_to_pixel(landmarks, L_EYE_B, w, h)
        rt = lm_to_pixel(landmarks, R_EYE_T, w, h)
        rb = lm_to_pixel(landmarks, R_EYE_B, w, h)

        left_h = abs(lt[1] - lb[1])
        right_h = abs(rt[1] - rb[1])
        eye_pixels = (left_h + right_h) / 2.0

        # mouth gap
        mt = lm_to_pixel(landmarks, MOUTH_TOP, w, h)
        mb = lm_to_pixel(landmarks, MOUTH_BOTTOM, w, h)
        mouth_pixels = abs(mb[1] - mt[1])

    # smoothing
    eh = state["eye_history"]
    mh = state["mouth_history"]
    eh.append(eye_pixels)
    mh.append(mouth_pixels)

    eye_avg = float(np.mean(eh)) if len(eh) else 0.0
    mouth_avg = float(np.mean(mh)) if len(mh) else 0.0

    # ---------------- calibration (manual trigger)
    if state.get("calibrating", False):
        if state["_calib_start"] is None:
            state["_calib_start"] = time.time()
            state["_calib_samples"] = []
            queue_audio("Calibration started")
        else:
            state["calib_elapsed"] = time.time() - state["_calib_start"]

        state["_calib_samples"].append(eye_pixels)

        if time.time() - state["_calib_start"] >= CALIBRATION_TIME:
            baseline = float(np.median(state["_calib_samples"])) if state["_calib_samples"] else 12.0
            thresh = baseline * 0.6
            state["baseline"] = baseline
            state["eye_thresh"] = thresh
            state["calibrating"] = False
            state["calib_elapsed"] = 0.0
            state["_calib_start"] = None
            state["_calib_samples"] = []
            queue_audio("Calibration complete")

    # ---------------- yawn detection
    if mouth_avg > YAWN_PHYS_PIXELS:
        if not state["_yawning"]:
            state["_yawning"] = True
            state["_yawn_start"] = time.time()
        else:
            if time.time() - state["_yawn_start"] >= YAWN_MIN_DURATION:
                if time.time() - state["_last_yawn_time"] >= YAWN_COOLDOWN:
                    state["yawns"] += 1
                    state["_last_yawn_time"] = time.time()
                    state["_yawning"] = False
                    queue_audio("Yawn detected")
    else:
        state["_yawning"] = False

    # ---------------- eye closure detection
    thresh = state.get("eye_thresh") or DEFAULT_EYE_THRESH
    if eye_avg < thresh:
        state["_closed_frames"] += 1
    else:
        state["_closed_frames"] = 0

    now = time.time()
    if state["_closed_frames"] >= CONSEC_FRAMES_RED:
        if now - state["_last_alert"] >= ALERT_INTERVAL:
            state["alerts"] += 1
            state["status"] = "DROWSY"
            state["_last_alert"] = now
            queue_audio("You look drowsy. Please take a break.")
    else:
        if state.get("calibrating"):
            state["status"] = "CALIBRATING"
        elif mouth_avg > YAWN_PHYS_PIXELS:
            state["status"] = "YAWNING"
        else:
            state["status"] = "ACTIVE"

    # update numeric stats
    state["eye_avg"] = eye_avg
    state["mouth_avg"] = mouth_avg

    # draw overlay (PIL)
    annotated = pil_image.copy()
    annotated = draw_overlay_pil(annotated, landmarks, w, h, state)

    return annotated

# buttons behavior
if start_btn:
    if img_file is None:
        st.warning("No image captured — click the camera widget first.")
    else:
        try:
            # read image into PIL
            bytes_data = img_file.getvalue()
            pil = Image.open(BytesIO(bytes_data)).convert("RGB")
            annotated = process_frame_pil(pil)
            img_slot.image(annotated, use_column_width=True)
        except Exception as e:
            st.error(f"Processing error: {e}")
            traceback.print_exc()

if calib_btn:
    # start manual calibration — will use subsequent snapshots to collect samples
    state["calibrating"] = True
    state["calib_elapsed"] = 0.0
    state["_calib_start"] = None
    state["_calib_samples"] = []
    st.info("Calibration started — take snapshots for the next few seconds.")

# auto-display last analysis if exists (nice UX)
if img_file and not start_btn:
    # show the captured image (no overlay) so user sees what they snapped
    img_slot.image(img_file, use_column_width=True)

# ----------------------------
# audio playback loop (main thread)
while not audio_queue.empty():
    msg = audio_queue.get()
    audio_bytes, mime = synthesize_audio_bytes(msg)
    if audio_bytes:
        st.audio(audio_bytes, format=mime)
    audio_queue.task_done()

# status + meta display
s_snapshot = dict(state)
status_slot.markdown(
    f"**Status:** {s_snapshot.get('status','-')} • "
    f"**Yawns:** {s_snapshot.get('yawns',0)} • "
    f"**Alerts:** {s_snapshot.get('alerts',0)}"
)

eye_val = s_snapshot.get("eye_avg") or 0.0
mouth_val = s_snapshot.get("mouth_avg") or 0.0
baseline = s_snapshot.get("baseline") or 0.0
thr = s_snapshot.get("eye_thresh") or DEFAULT_EYE_THRESH

if s_snapshot.get("calibrating"):
    meta_slot.info(f"Calibrating… {s_snapshot.get('calib_elapsed',0):.1f}/{CALIBRATION_TIME}s")
else:
    meta_slot.write(
        f"Eye(px): {eye_val:.1f} | "
        f"Mouth(px): {mouth_val:.1f} | "
        f"Baseline: {baseline:.1f} | "
        f"Threshold(px): {thr:.1f}"
    )

# helpful footer
st.caption("Tip: click 'Take Snapshot' to analyze the frame. Use 'Calibrate' then take snapshots for accurate eye thresholding.")

