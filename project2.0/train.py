import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import os

sequence_length = 30  # Should match previous steps
save_dir = "gesture_data"

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_test_model():
    # Load preprocessed data
    X = np.load(os.path.join(save_dir, "X.npy"))
    y = np.load(os.path.join(save_dir, "y.npy"))

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train LSTM model
    model = build_lstm_model((sequence_length, X_train.shape[2]), len(np.unique(y)))
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Save model
    model.save('gesture_lstm_model.h5')

if __name__ == "__main__":
    train_and_test_model()
