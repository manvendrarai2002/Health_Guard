import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load data
print("Loading dataset...")
try:
    df = pd.read_csv('data/medical_data.csv')
except FileNotFoundError:
    print("Couldn't find data/medical_data.csv. Run generate_data.py first.")
    exit(1)

X = df.drop('Disease', axis=1)
y = df['Disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model (before SMOTE/tuning)
print("\n--- Baseline Model (Before Tuning) ---")
clf_base = RandomForestClassifier(random_state=42)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)
print(classification_report(y_test, y_pred_base))

# Apply SMOTE
print("\n--- Applying SMOTE ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Original training size: {X_train.shape[0]}, Resampled size: {X_train_smote.shape[0]}")

# Tune with GridSearchCV
print("\n--- Tuning with GridSearchCV ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
# Total combinations = 3 * 4 * 3 = 36

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

start_time = time.time()
grid_search.fit(X_train_smote, y_train_smote)
end_time = time.time()
print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds")

best_clf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Validation
print("\n--- Validating Final Model ---")
# Metrics
cv_scores = cross_val_score(best_clf, X_train_smote, y_train_smote, cv=5)
print(f"5-Fold Cross-Validation Score: {cv_scores.mean():.2f}")

y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.1f}%")

print("\nClassification Report (Final):")
print(classification_report(y_test, y_pred))

# Latency check
print("\n--- Checking Prediction Latency ---")
import time
start_lat = time.time()
# Run a small loop to estimate average prediction time
for _ in range(100):
   _ = best_clf.predict(X_test.iloc[0:1])
end_lat = time.time()
avg_latency_ms = (end_lat - start_lat) / 100 * 1000
print(f"Average Prediction Latency: {avg_latency_ms:.2f} ms")

# Save model
import os
os.makedirs('models', exist_ok=True)
joblib.dump(best_clf, 'models/model.pkl')
print("\nSaved model to models/model.pkl")
