import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import pyautogui  # To control media actions like volume or play/pause

# Load the trained model
clf = joblib.load('gesture_classifier.pkl')

playerstate = ''

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

#print on screen
screen_text = ''

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
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks).flatten().reshape(1, -1)
            
            
            if landmarks.size == 0 | np.isnan(landmarks).any():
                predicted_gesture = 'none'
            else:
                # Predict gesture
                predicted_gesture = clf.predict(landmarks)[0]
                        
            # Perform media control based on the predicted gesture
            if predicted_gesture == 'stop':
                pyautogui.press('stop')
                screen_text = 'stop'
            if predicted_gesture == 'volumeup':
                pyautogui.press(['ctrl', 'up'])
                screen_text = 'up'
                time.sleep(0.1)
            elif predicted_gesture == 'volumedown':
                pyautogui.press(['ctrl', 'down'])
                time.sleep(0.1)
                screen_text = 'down'
            elif predicted_gesture == 'forward':
                pyautogui.press(['ctrl', 'right'])
                time.sleep(0.1)
                screen_text = 'forward'
            elif predicted_gesture == 'backward':
                pyautogui.press(['ctrl', 'left'])
                time.sleep(0.1)
                screen_text = 'back'
            elif predicted_gesture == 'pause':
                pyautogui.press('playpause')
                screen_text = 'playpause'
                time.sleep(0.7)
                screen_text = 'playpause'
            elif predicted_gesture == 'play':
                pyautogui.press('playpause')
                time.sleep(0.7)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Show the predicted gesture on the screen
            if predicted_gesture:
                cv2.putText(frame, f'Gesture: {screen_text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
