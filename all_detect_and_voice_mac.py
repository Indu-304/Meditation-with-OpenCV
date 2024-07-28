from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from fer import FER
import subprocess
import time
import threading
import pyaudio
import numpy as np

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def speak(text):
    subprocess.call(["say", text])

# to ensure warnings are spoken with a cooldown period
def speak_warning(text, cooldown_period, last_speak_time, speak_thread):
    current_time = time.time()
    if current_time - last_speak_time > cooldown_period and (speak_thread is None or not speak_thread.is_alive()):
        speak_thread = threading.Thread(target=speak, args=(text,))
        speak_thread.start()
        return current_time, speak_thread
    return last_speak_time, speak_thread

def get_noise_level(stream, chunk):
    data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.int16)
    peak = np.abs(data).max()
    noise_level = 20 * np.log10(peak) if peak > 0 else 0
    return noise_level

def put_text_with_shadow(frame, text, x, y, font_scale, color, thickness):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    rect_x1 = x
    rect_y1 = y - text_size[1] - 10
    rect_x2 = x + text_size[0] + 10
    rect_y2 = y + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return frame

# Initialize audio settings
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Initialize face detection and landmark prediction
thresh = 0.20
frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/Users/Indu/Desktop/Programs/OpenCV/Meditation_support/models/shape_predictor_68_face_landmarks.dat")

# Indices for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize video capture
cap = cv2.VideoCapture(0)
flag = 0

# Initialize emotion detector and movement tracking variables
emotion_detector = FER()
prev_position = None
movement_threshold = 50

# Initialize cooldown periods for warnings
last_movement_warning_time = 0
movement_cooldown_period = 5
last_emotion_warning_time = 0
emotion_cooldown_period = 5

speak_thread = None

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check emotion
        emotion, emotion_score = emotion_detector.top_emotion(frame)
        if emotion:
            emotion_text = f"Emotion: {emotion} ({emotion_score:.2f})"
            put_text_with_shadow(frame, emotion_text, 10, 30, 1, (255, 255, 255), 1)
        if emotion and emotion_score > 0.7 and emotion not in ["happy", "neutral"]:
            warning_text = "Please smile"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] - text_size[1]) / 2) + 100
            put_text_with_shadow(frame, warning_text, text_x, text_y, 1, (255, 255, 255), 2)
            if ear < thresh:
                last_emotion_warning_time, speak_thread = speak_warning(warning_text, emotion_cooldown_period, last_emotion_warning_time, speak_thread)

        # Check eye aspect ratio for drowsiness
        if ear > thresh:
            flag += 1
            if flag >= frame_check:
                text = "CLOSE YOUR EYES!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int((frame.shape[0] + text_size[1]) / 2)
                put_text_with_shadow(frame, text, text_x, text_y, 1, (255, 255, 255), 2)
        else:
            flag = 0
           
        # Check for movement
        current_position = (subject.left(), subject.top(), subject.width(), subject.height())
        if prev_position is not None:
            movement = distance.euclidean(current_position, prev_position)
            if movement > movement_threshold:
                movement_text = "Please sit still!"
                text_size = cv2.getTextSize(movement_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int((frame.shape[0] - text_size[1]) / 2) - 50
                put_text_with_shadow(frame, movement_text, text_x, text_y, 1, (255, 255, 255), 2)
                if ear < thresh: # If eyes are closed, then voice warning
                    last_movement_warning_time, speak_thread = speak_warning("Please sit still!", movement_cooldown_period, last_movement_warning_time, speak_thread)
        
        prev_position = current_position

    # Get noise level and display it
    noise_level = get_noise_level(stream, chunk=1024)
    noise_text = f"Noise Level: {noise_level:.2f} dB"
    put_text_with_shadow(frame, noise_text, frame.shape[1] - 300, frame.shape[0] - 10, 1, (255, 255, 255), 1)
    if noise_level > 80:
        put_text_with_shadow(frame, "Noise is greater than recommended level", 10, frame.shape[0] - 10, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        stream.stop_stream()
        stream.close()
        p.terminate()
        break
