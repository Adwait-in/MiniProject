from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the collected gesture data
data = pd.read_csv('hand_gesture_data_combined.csv')

# Split features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
import joblib
joblib.dump(clf, 'gesture_classifier.pkl')

# n_estimators: The number of trees in the forest. 
#     A larger number generally improves accuracy but increases computation time.
# random_state: seed for the random number generator to ensure repetability
# max_depth: The maximum depth of each tree. Limiting depth helps control overfitting.
# min_samples_split: The minimum number of samples required to split a node.
# max_features: The number of features to consider when looking for 
#       the best split. Lower values reduce correlation between trees.
# bootstrap: Whether to use bootstrapped samples when building trees.
