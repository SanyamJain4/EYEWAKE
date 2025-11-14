import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Streamlit UI setup
st.set_page_config(page_title="EYEWAKE - Drowsiness Detection", layout="wide")
st.title("👁️ EYEWAKE: Real-Time Alertness Monitor")
st.markdown("Monitor your eye and mouth activity in real-time using your webcam.")

# Buttons for control
start = st.button("▶️ Start Detection")
stop = st.button("⏹️ Stop Detection")

# Display webcam feed and status
frame_window = st.image([])
status_text = st.empty()

# Eye and yawn state calculation
def detect_states(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    state = "No Face Detected"
    color = (255, 255, 255)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Eye coordinates (left eye)
            left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
            left_eye_bottom = np.array([landmarks[145].x * w, landmarks[145].y * h])
            eye_distance = np.linalg.norm(left_eye_top - left_eye_bottom)

            # Mouth coordinates
            mouth_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
            mouth_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
            mouth_distance = np.linalg.norm(mouth_top - mouth_bottom)

            if eye_distance < 5:
                state = "😴 Drowsy"
                color = (0, 0, 255)
            elif mouth_distance > 25:
                state = "😮 Yawning"
                color = (0, 165, 255)
            else:
                state = "😊 Active"
                color = (0, 255, 0)

            # Draw status on frame
            cv2.putText(frame, state, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    return frame, state


# Detection logic
if start and not stop:
    cap = cv2.VideoCapture(0)
    status_text.info("Starting detection... Press ⏹️ Stop Detection to end.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("No camera input detected.")
            break

        frame = cv2.flip(frame, 1)
        frame, state = detect_states(frame)
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status_text.markdown(f"### Current State: **{state}**")

        if stop:
            break

    cap.release()
    status_text.success("Detection stopped.")
