import numpy as np
import os

gesture_names = ["left_swipe", "right_swipe"]  # List of all gesture names "zoomIn", "zoomOut"
sequence_length = 30  # Should match the one used in data collection
save_dir = "gesture_data"

def combine_and_preprocess():
    X, y = [], []

    for label, gesture_name in enumerate(gesture_names):
        gesture_sequences = np.load(os.path.join(save_dir, f"{gesture_name}.npy"))
        X.extend(gesture_sequences)
        y.extend([label] * len(gesture_sequences))

    X = np.array(X)
    y = np.array(y)

    # Normalize data
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X, y

if __name__ == "__main__":
    X, y = combine_and_preprocess()
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)