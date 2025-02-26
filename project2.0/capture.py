import cv2
import mediapipe as mp
import numpy as np
import os
import sys

gesture_name = "NA"
if len(sys.argv) < 2:
    print("Please provide a gesture label")
    exit()
else:
    # gesture label for data collection
    gesture_name = sys.argv[1]
    print(f"Capturing for gesture - {gesture_name}!")

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks
hands = mp_hands.Hands(max_num_hands=1)  # Process only 1 hand at a time

# Define parameters for capturing
sequence_length = 30  # Number of frames per gesture
save_dir = "gesture_data"  # Directory to save gesture data
hand_landmark_sequences = []
current_sequence = []

def collect_gesture_data():
    global current_sequence, hand_landmark_sequences

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Process only the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Collect landmarks for the sequence
            current_sequence.append(landmarks)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the sequence length is complete
            if len(current_sequence) == sequence_length:
                hand_landmark_sequences.append(current_sequence)
                current_sequence = []

        # Display progress
        progress_text = f"Frames Captured: {len(current_sequence)}/{sequence_length}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with landmarks
        cv2.imshow('Collecting Gesture Data', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save collected sequences
    np.save(os.path.join(save_dir, f"{gesture_name}.npy"), np.array(hand_landmark_sequences))

if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    collect_gesture_data()
