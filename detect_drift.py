from pathlib import Path
import pandas as pd
from scipy.stats import ks_2samp

# Step 1: Define file paths
REFERENCE_PATH = Path("data") / "reference_data.csv"
INCOMING_PATH = Path("data") / "incoming_data.csv"

# Step 2: Load reference and incoming data
reference_data = pd.read_csv(REFERENCE_PATH)
incoming_data = pd.read_csv(INCOMING_PATH)

# Step 3: Select only input features
incoming_data = incoming_data.drop(
    columns=["actual_quality"],
    errors="ignore"
)

# Step 4: Set drift detection threshold
P_VALUE_THRESHOLD = 0.05

print("\n========== DATA DRIFT DETECTION ==========")

# Step 5: Compare every feature
for column in reference_data.columns:

    reference_values = reference_data[column]
    incoming_values = incoming_data[column]

    # Perform Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(
        reference_values,
        incoming_values
    )

    # Decide whether drift is detected
    if p_value < P_VALUE_THRESHOLD:
        status = "DRIFT DETECTED"
    else:
        status = "NO DRIFT"

    # Display results
    print("\nFeature:", column)
    print("KS Statistic:", round(statistic, 4))
    print("P-value:", round(p_value, 4))
    print("Status:", status)

print("\n========== DRIFT CHECK COMPLETED ==========")