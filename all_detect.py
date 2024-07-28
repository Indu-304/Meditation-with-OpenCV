from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from fer import FER

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.20 
frame_check = 10 #number of frames with eyes open
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/Users/Indu/Desktop/Programs/OpenCV/Meditation_support/models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0

emotion_detector = FER()
prev_position = None
movement_threshold = 30  # Adjust this threshold based on your requirements

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
        
        if ear > thresh:
            flag += 1
            if flag >= frame_check:
                text = "***CLOSE YOUR EYES!***"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int((frame.shape[0] + text_size[1]) / 2)
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 255), 2)
        else:
            flag = 0
        
        emotion, score = emotion_detector.top_emotion(frame)
        if emotion:
            emotion_text = f"Emotion: {emotion} ({score:.2f})"
            cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check for movement
        current_position = (subject.left(), subject.top(), subject.width(), subject.height())
        if prev_position is not None:
            movement = distance.euclidean(current_position, prev_position)
            if movement > movement_threshold:
                movement_text = "Please sit still!"
                text_size = cv2.getTextSize(movement_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int((frame.shape[0] - text_size[1]) / 2) - 50
                cv2.putText(frame, movement_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        prev_position = current_position

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
