
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load dataset
DATA_PATH = Path("data") / "winequality-red.csv"
df = pd.read_csv(DATA_PATH, sep=";")

# Step 2: Create quality categories
def categorize_quality(quality):
    if quality <= 4:
        return "Low"
    elif quality <= 6:
        return "Medium"
    else:
        return "High"

df["quality_category"] = df["quality"].apply(categorize_quality)

# Step 3: Separate features and target
X = df.drop(columns=["quality", "quality_category"])
y = df["quality_category"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\n--- MODEL EVALUATION ---")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]))

# Step 9: Save the trained model
MODEL_PATH = Path("models") / "wine_quality_model.pkl"
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully at:", MODEL_PATH)

# Step 10: Save reference data for drift monitoring
REFERENCE_PATH = Path("data") / "reference_data.csv"
X_train.to_csv(REFERENCE_PATH, index=False)

# Step 11: Save labeled test data for monitoring experiments
TEST_PATH = Path("data") / "test_data.csv"

test_data = X_test.copy()
test_data["actual_quality"] = y_test

test_data.to_csv(TEST_PATH, index=False)

print("\n--- MONITORING DATA SAVED ---")
print("Reference data:", REFERENCE_PATH)
print("Test data:", TEST_PATH)

