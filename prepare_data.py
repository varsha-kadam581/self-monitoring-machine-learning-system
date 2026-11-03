
from pathlib import Path
import pandas as pd

# Step 1: Load the original dataset
DATA_PATH = Path("data") / "winequality-red.csv"

df = pd.read_csv(DATA_PATH, sep=";")

# Step 2: Create a new target column
def categorize_quality(quality):
    if quality <= 4:
        return "Low"
    elif quality <= 6:
        return "Medium"
    else:
        return "High"

df["quality_category"] = df["quality"].apply(categorize_quality)

# Step 3: Separate input features and target
X = df.drop(columns=["quality", "quality_category"])
y = df["quality_category"]

# Step 4: Display the results
print("\n--- INPUT FEATURES ---")
print(X.head())

print("\n--- TARGET VALUES ---")
print(y.head())

print("\n--- TARGET DISTRIBUTION ---")
print(y.value_counts())

print("\nNumber of input features:", X.shape[1])
print("Number of records:", X.shape[0])
