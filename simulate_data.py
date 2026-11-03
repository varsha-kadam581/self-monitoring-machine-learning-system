from pathlib import Path
import pandas as pd

# Step 1: Load the test dataset
TEST_PATH = Path("data") / "test_data.csv"

df = pd.read_csv(TEST_PATH)

# Step 2: Create a copy of the test data
incoming_data = df.copy()

# Step 3: Simulate data drift
# Increase alcohol values by 30%
incoming_data["alcohol"] = incoming_data["alcohol"] * 1.30

# Step 4: Save simulated incoming data
OUTPUT_PATH = Path("data") / "incoming_data.csv"

incoming_data.to_csv(OUTPUT_PATH, index=False)

# Step 5: Display results
print("\n--- SIMULATED DATA CREATED ---")
print("Original alcohol average:", round(df["alcohol"].mean(), 2))
print("Incoming alcohol average:", round(incoming_data["alcohol"].mean(), 2))

print("\nSaved file:", OUTPUT_PATH)
print("Number of incoming records:", len(incoming_data))