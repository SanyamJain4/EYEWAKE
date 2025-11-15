# streamlit_app.py
# Eyewake — streamlit-webrtc version (browser camera -> server-side MediaPipe)
import streamlit as st
import time
import queue
import uuid
import os
import tempfile
import traceback

import cv2
import numpy as np
import mediapipe as mp

# webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Optional audio engines
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# -------------------------
# Constants (copied from your original)
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

# -------------------------
# Shared state (thread-safe-ish since transformer and main thread run separately)
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

# audio queue (main thread will play)
audio_queue = queue.Queue()

def synthesize_audio_bytes(text):
    """Try pyttsx3 → gTTS fallback → return (bytes, mime) or (None, None)."""
    uid = uuid.uuid4().hex

    # pyttsx3 offline
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

    # gTTS fallback
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

# -------------------------
# Helper to convert normalized landmarks to pixels
def extract_point(lm, idx, w, h):
    p = lm[idx]
    return (int(p.x * w), int(p.y * h))

# -------------------------
# Video transformer: runs in worker thread created by streamlit-webrtc
class MediaPipeTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_history = []
        self.mouth_history = []
        self.closed_frames = 0

        self.yawning = False
        self.yawn_start = 0
        self.last_yawn_time = 0
        self.last_alert = 0

        self.calib_start = None
        self.calib_samples = []

    def transform(self, frame):
        """
        frame: av.VideoFrame wrapper from streamlit-webrtc
        Should return a numpy ndarray (BGR) to be shown in the browser.
        """
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        eye_pixels = 0.0
        mouth_pixels = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            lt = extract_point(lm, L_EYE_T, w, h)
            lb = extract_point(lm, L_EYE_B, w, h)
            rt = extract_point(lm, R_EYE_T, w, h)
            rb = extract_point(lm, R_EYE_B, w, h)

            left_h = abs(lt[1] - lb[1])
            right_h = abs(rt[1] - rb[1])
            eye_pixels = (left_h + right_h) / 2.0

            mt = extract_point(lm, MOUTH_TOP, w, h)
            mb = extract_point(lm, MOUTH_BOTTOM, w, h)
            mouth_pixels = abs(mb[1] - mt[1])

            # draw markers
            for idx in (L_EYE_T, L_EYE_B, R_EYE_T, R_EYE_B, MOUTH_TOP, MOUTH_BOTTOM):
                p = extract_point(lm, idx, w, h)
                cv2.circle(img, p, 2, (0,255,0), -1)

        # smoothing (circular buffer like)
        self.eye_history.append(eye_pixels)
        if len(self.eye_history) > EYE_HISTORY_LEN:
            self.eye_history.pop(0)

        self.mouth_history.append(mouth_pixels)
        if len(self.mouth_history) > EYE_HISTORY_LEN:
            self.mouth_history.pop(0)

        eye_avg = float(np.mean(self.eye_history)) if self.eye_history else 0.0
        mouth_avg = float(np.mean(self.mouth_history)) if self.mouth_history else 0.0

        # calibration (the main thread sets state["calibrating"]=True when user presses Calibrate)
        if state.get("calibrating", False):
            if self.calib_start is None:
                self.calib_start = time.time()
                self.calib_samples = []
                queue_audio("Calibration started")
            else:
                state["calib_elapsed"] = time.time() - self.calib_start

            self.calib_samples.append(eye_pixels)

            if time.time() - self.calib_start >= CALIBRATION_TIME:
                baseline = float(np.median(self.calib_samples)) if self.calib_samples else 12.0
                thresh = baseline * 0.6

                state["baseline"] = baseline
                state["eye_thresh"] = thresh
                state["calibrating"] = False
                state["calib_elapsed"] = 0.0

                self.calib_start = None
                self.calib_samples = []
                queue_audio("Calibration complete")

        # yawn detection
        if mouth_avg > YAWN_PHYS_PIXELS:
            if not self.yawning:
                self.yawning = True
                self.yawn_start = time.time()
            else:
                if time.time() - self.yawn_start >= YAWN_MIN_DURATION:
                    if time.time() - self.last_yawn_time >= YAWN_COOLDOWN:
                        state["yawns"] += 1
                        self.last_yawn_time = time.time()
                        self.yawning = False
                        queue_audio("Yawn detected")
        else:
            self.yawning = False

        # eye closure detection
        thresh = state.get("eye_thresh") or DEFAULT_EYE_THRESH

        if eye_avg < thresh:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        now = time.time()

        if self.closed_frames >= CONSEC_FRAMES_RED:
            if now - self.last_alert >= ALERT_INTERVAL:
                state["alerts"] += 1
                state["status"] = "DROWSY"
                self.last_alert = now
                queue_audio("You look drowsy. Please take a break.")
        else:
            if state.get("calibrating", False):
                state["status"] = "CALIBRATING"
            elif mouth_avg > YAWN_PHYS_PIXELS:
                state["status"] = "YAWNING"
            else:
                state["status"] = "ACTIVE"

        # update shared state values
        state["eye_avg"] = eye_avg
        state["mouth_avg"] = mouth_avg

        # overlay status text
        overlay = img
        cv2.putText(overlay, f"Status: {state['status']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(overlay,
                    f"Eye(px): {eye_avg:.1f}  Thresh(px): {(state.get('eye_thresh') or DEFAULT_EYE_THRESH):.1f}",
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

        # return BGR image
        return overlay

# -------------------------
# Streamlit UI
st.set_page_config(page_title="Eyewake", layout="centered")
st.title("🕶️ EYEWAKE — Pixel-based Detection (webrtc)")

with st.sidebar:
    st.header("Camera / Settings")
    # camera index is not used in browser flow, but keep UI for parity
    cam_idx = st.number_input("Camera Index (ignored for browser camera)", min_value=0, max_value=5, value=0)

col1, col2, col3 = st.columns(3)

webrtc_ctx = None

with col1:
    start_btn = st.button("Start")

with col2:
    calibrate_btn = st.button("Calibrate")

with col3:
    stop_btn = st.button("Stop")

# start/stop logic (webrtc_streamer starts when called; we rely on button to show or not show streamer)
if start_btn:
    # start the webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="eyewake",
        video_transformer_factory=MediaPipeTransformer,
        rtc_configuration={},
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    state["status"] = "STARTED"
    queue_audio("Detection started")

if calibrate_btn:
    if webrtc_ctx and webrtc_ctx.state.playing:
        state["calibrating"] = True
    else:
        st.warning("Start detection first.")

if stop_btn:
    # there is no direct stop API except stopping the player in the browser; set status
    state["status"] = "STOPPED"
    queue_audio("Stopping")
    # inform user to stop the stream in UI
    st.info("Click the stop button in the little video player (top-right) to stop the camera.")

# Display state & audio playback loop
status_slot = st.empty()
meta_slot = st.empty()

# loop to update UI values and play queued audio
try:
    while True:
        # Play queued audio (main thread)
        while not audio_queue.empty():
            msg = audio_queue.get()
            audio_bytes, mime = synthesize_audio_bytes(msg)
            if audio_bytes:
                st.audio(audio_bytes, format=mime)
            audio_queue.task_done()

        # Show state
        s = dict(state)  # snapshot
        status_slot.markdown(
            f"**Status:** {s.get('status','-')} • "
            f"**Yawns:** {s.get('yawns',0)} • "
            f"**Alerts:** {s.get('alerts',0)}"
        )

        eye_val = s.get("eye_avg") or 0.0
        mouth_val = s.get("mouth_avg") or 0.0
        baseline = s.get("baseline") or 0.0
        thr = s.get("eye_thresh") or DEFAULT_EYE_THRESH

        if s.get("calibrating"):
            meta_slot.info(f"Calibrating… {s.get('calib_elapsed',0):.1f}/{CALIBRATION_TIME}s")
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
