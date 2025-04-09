import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import deque

# Constants
FRAME_RATE = 30  # Assumed frame rate
DURATION_THRESHOLD = 5  # Duration in seconds
HISTORY_SIZE = 15  # Number of frames to keep in history for stabilization

# Initialize state counters
sleeping_counter = 0

listening_counter = 0

# Initialize state history deque
state_history = deque(maxlen=HISTORY_SIZE)

# Preprocessing function for low light conditions
def preprocess_image(image):
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)

    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_image

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_activity(frame):
    global sleeping_counter, using_mobile_counter, listening_counter, state_history

    preprocessed_frame = preprocess_image(frame)
    gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    state_detected = None

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            ear_values = []
            for (ex, ey, ew, eh) in eyes:
                eye = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_aspect_ratio_value = eye_aspect_ratio(eye)
                ear_values.append(eye_aspect_ratio_value)
            
            if len(ear_values) >= 2:
                average_ear = sum(ear_values) / len(ear_values)
                if average_ear < 0.25:
                    state_detected = 'Sleeping'
                    cv2.putText(frame, 'Sleeping', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    state_detected = 'Listening'
                    cv2.putText(frame, 'Listening', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif len(eyes) == 0:
            state_detected = 'Sleeping'
            cv2.putText(frame, 'Sleeping', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    # Append the detected state to the history
    if state_detected:
        state_history.append(state_detected)

    # Determine the most frequent state in the history
    if len(state_history) == HISTORY_SIZE:
        most_common_state = max(set(state_history), key=state_history.count)

        # Update counters based on the most common state
        if most_common_state == 'Sleeping':
            sleeping_counter += 1
            using_mobile_counter = 0
            listening_counter = 0
        elif most_common_state == 'Listening':
            listening_counter += 1
            sleeping_counter = 0
            using_mobile_counter = 0

        # Check if the state persists for more than the threshold duration
        if sleeping_counter > DURATION_THRESHOLD * FRAME_RATE:
            # Display popup for sleeping
            while True:
                cv2.putText(frame, 'Warning: Sleeping for more than 5 seconds!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 20, 150), 2)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('o'):  # Wait for 'o' key to continue
                    break
            sleeping_counter = 0  # Reset the counter after triggering the popup

    
    return frame

# Load pre-trained face and eye cascades (ensure you have the XML files)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Example usage with a video feed (adjust the source as needed)
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_activity(frame)

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
