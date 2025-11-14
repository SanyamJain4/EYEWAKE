# ================================
# Eyewake Local Server (Windows-safe)
# No eventlet, no monkey_patch
# ================================

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# -------------------------
# Flask + SocketIO (threading mode)
# -------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -------------------------
# Global states
# -------------------------
camera_index = 0
running = True
stop_requested = False
calibrating = False

CALIB_DURATION = 5.0
calib_samples = []
calib_start_time = None

# Eye thresholds (will be calibrated)
EYE_THRESH = 0.20
YAWN_THRESH = 0.35

eye_history = deque(maxlen=10)
mouth_history = deque(maxlen=10)

mp_face = mp.solutions.face_mesh

# -------------------------
# Helpers
# -------------------------
def extract_point(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h)

def detection_generator():
    global running, stop_requested, calibrating, calib_start_time, calib_samples, EYE_THRESH

    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 640)
    cap.set(4, 480)

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    closed_frames = 0
    yawning = False
    yawns = 0
    alerts = 0
    last_alert = 0

    while running and not stop_requested:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        h, w = frame.shape[:2]
        eye_val = 0
        mouth_val = 0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Eye landmarks
            lt = extract_point(lm, 159, w, h)
            lb = extract_point(lm, 145, w, h)
            rt = extract_point(lm, 386, w, h)
            rb = extract_point(lm, 374, w, h)

            left = abs(lt[1] - lb[1]) / h
            right = abs(rt[1] - rb[1]) / h
            eye_val = (left + right) / 2

            # Mouth
            mt = extract_point(lm, 13, w, h)
            mb = extract_point(lm, 14, w, h)
            mouth_val = abs(mb[1] - mt[1]) / h

        eye_history.append(eye_val)
        mouth_history.append(mouth_val)

        eye_avg = float(np.mean(eye_history))
        mouth_avg = float(np.mean(mouth_history))

        # ---- Calibration ----
        if calibrating:
            if calib_start_time is None:
                calib_start_time = time.time()
            calib_samples.append(eye_avg)

            elapsed = time.time() - calib_start_time
            cv2.putText(frame, f"Calibrating {elapsed:.1f}/{CALIB_DURATION}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if elapsed >= CALIB_DURATION:
                baseline = float(np.mean(calib_samples))
                EYE_THRESH = baseline * 0.55
                calibrating = False
                calib_samples = []
                calib_start_time = None
                socketio.emit("calibrated", {"baseline": baseline, "threshold": EYE_THRESH})

        # ---- Yawn detection ----
        if mouth_avg > YAWN_THRESH and not yawning:
            yawning = True
            yawns += 1
            socketio.emit("status", {"status": "YAWNING", "yawns": yawns})

        if mouth_avg <= YAWN_THRESH:
            yawning = False

        # ---- Eye closure detection ----
        if eye_avg < EYE_THRESH and eye_avg > 0:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames > 20:
            # Drowsy alert
            if time.time() - last_alert > 2:
                alerts += 1
                last_alert = time.time()
                socketio.emit("status", {"status": "DROWSY", "alerts": alerts})
        else:
            socketio.emit("status", {"status": "ACTIVE", "alerts": alerts})

        # Convert to JPEG for streaming
        ret, buf = cv2.imencode(".jpg", frame)
        frame_bytes = buf.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

    cap.release()
    socketio.emit("stopped", {})

# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(detection_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------------
# SocketIO events
# -------------------------
@socketio.on("calibrate")
def do_calibrate(_):
    global calibrating, calib_samples, calib_start_time
    calibrating = True
    calib_samples = []
    calib_start_time = None
    socketio.emit("calibration_started", {"duration": CALIB_DURATION})

@socketio.on("stop")
def stop(_):
    global stop_requested
    stop_requested = True

# -------------------------
# Start Server
# -------------------------
if __name__ == "__main__":
    print("Eyewake server running at http://127.0.0.1:5001")
    socketio.run(app, host="127.0.0.1", port=5000)

