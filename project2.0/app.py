import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

# Initialize Mediapipe and load trained model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
model = load_model('gesture_lstm_model.h5')

sequence_length = 30
current_sequence = []

gesture_mapping = {0: 'left_swipe', 1: 'right_swipe', 2: 'zoomIn', 3: 'zoomOut'}

def control_computer(gesture):
    if gesture == 'left_swipe':
        # pyautogui.hotkey('alt', 'tab')  # Switch between applications
        print('Left Swipe')
    elif gesture == 'right_swipe':
        # pyautogui.hotkey('alt', 'shift', 'tab')
        print('Right Swipe')
    elif gesture == 'zoomIn':
        # pyautogui.hotkey('ctrl', '+')  # Zoom in
        print('Zoom In')
    elif gesture == 'zoomOut':
        # pyautogui.hotkey('ctrl', '+')  # Zoom in
        print('zoom Out')

def gesture_recognition_app():
    global current_sequence
    cap = cv2.VideoCapture(0)
    detected_gesture = ""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                current_sequence.append(landmarks)

                if len(current_sequence) == sequence_length:
                    sequence_array = np.array(current_sequence)
                    sequence_array = np.expand_dims(sequence_array, axis=0)
                    sequence_array = (sequence_array - np.min(sequence_array)) / (np.max(sequence_array) - np.min(sequence_array))

                    prediction = model.predict(sequence_array)
                    predicted_label = np.argmax(prediction)
                    gesture = gesture_mapping[predicted_label]

                    detected_gesture = gesture
                    control_computer(gesture)

                    current_sequence = []

                    if detected_gesture:
                        cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display
        cv2.imshow('Gesture Control App', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_recognition_app()
