
from pathlib import Path
import pandas as pd

# Step 1: Find the dataset file
DATA_PATH = Path("data") / "winequality-red.csv"

# Step 2: Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH.resolve()}"
    )

# Step 3: Read the CSV file
df = pd.read_csv(DATA_PATH, sep=";")

# Step 4: Display basic information
print("\n--- DATASET LOADED SUCCESSFULLY ---")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Step 5: Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Step 6: Display the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# Step 7: Check data types and missing values
print("\n--- DATASET INFORMATION ---")
df.info()

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 8: Show basic statistics
print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())
# Step 9: Check the distribution of wine quality
print("\n--- WINE QUALITY DISTRIBUTION ---")
print(df["quality"].value_counts().sort_index())
