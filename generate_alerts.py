from pathlib import Path
import pandas as pd
import joblib

from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

# Step 1: Define file paths
REFERENCE_PATH = Path("data") / "reference_data.csv"
INCOMING_PATH = Path("data") / "incoming_data.csv"
TEST_PATH = Path("data") / "test_data.csv"
MODEL_PATH = Path("models") / "wine_quality_model.pkl"

# Step 2: Load files
reference_data = pd.read_csv(REFERENCE_PATH)
incoming_data = pd.read_csv(INCOMING_PATH)
test_data = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

# Step 3: Check data drift
P_VALUE_THRESHOLD = 0.05
drifted_features = []

incoming_features = incoming_data.drop(
    columns=["actual_quality"],
    errors="ignore"
)

for column in reference_data.columns:
    statistic, p_value = ks_2samp(
        reference_data[column],
        incoming_features[column]
    )

    if p_value < P_VALUE_THRESHOLD:
        drifted_features.append(column)

# Step 4: Check model performance
X_test = test_data.drop(columns=["actual_quality"])
y_actual = test_data["actual_quality"]

y_predicted = model.predict(X_test)

accuracy = accuracy_score(y_actual, y_predicted)

# Step 5: Generate alerts
alerts = []

if drifted_features:
    alerts.append(
        "DATA DRIFT ALERT: Drift detected in "
        + ", ".join(drifted_features)
    )

if accuracy < 0.80:
    alerts.append(
        f"PERFORMANCE ALERT: Model accuracy is "
        f"{accuracy * 100:.2f}%"
    )

# Step 6: Display alert results
print("\n========== AUTOMATED ALERT SYSTEM ==========")

if alerts:
    for alert in alerts:
        print("[WARNING]", alert)
else:
    print("[HEALTHY] No alerts detected.")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nDrifted Features:", drifted_features)

print("\n========== ALERT CHECK COMPLETED ==========")