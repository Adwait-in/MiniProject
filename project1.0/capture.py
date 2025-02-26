import cv2
import numpy as np
import pandas as pd
import mediapipe as mp # google hand landmarking algorithm library
import sys


gesture_label = "stop"
if len(sys.argv) < 2:
    print("Please provide a gesture label")
    exit()
else:
    # gesture label for data collection
    gesture_label = sys.argv[1]
    print(f"Capturing for gesture - {gesture_label}!")

# setup MediaPipe Hand landmarking 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Used OpenCV for video capture from webcam
cap = cv2.VideoCapture(0)

# temp store collected data
landmark_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
      
    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and extract hand landmarks
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks and append to data list
            landmarks = []
            for landmark in hand_landmarks.landmark:
              landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks).flatten()  # Flatten into a single array
            landmark_data.append([gesture_label] + landmarks.tolist())

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

# Convert collected data to a pandas DataFrame and save as CSV
columns = ['label'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
df = pd.DataFrame(landmark_data, columns=columns)
df.to_csv(f'hand_gesture_data_{gesture_label}.csv', index=False)
