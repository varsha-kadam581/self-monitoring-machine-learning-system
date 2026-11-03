from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Define file paths
MODEL_PATH = Path("models") / "wine_quality_model.pkl"
TEST_PATH = Path("data") / "test_data.csv"

# Step 2: Load the trained model
model = joblib.load(MODEL_PATH)

# Step 3: Load test data
test_data = pd.read_csv(TEST_PATH)

# Step 4: Separate features and actual labels
X_test = test_data.drop(columns=["actual_quality"])
y_actual = test_data["actual_quality"]

# Step 5: Generate model predictions
y_predicted = model.predict(X_test)

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_actual, y_predicted)

# Step 7: Display monitoring results
print("\n========== MODEL PERFORMANCE MONITORING ==========")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_actual,
        y_predicted,
        zero_division=0
    )
)

print("\n--- CONFUSION MATRIX ---")
print(
    confusion_matrix(
        y_actual,
        y_predicted,
        labels=["Low", "Medium", "High"]
    )
)

print("\n========== MONITORING COMPLETED ==========")