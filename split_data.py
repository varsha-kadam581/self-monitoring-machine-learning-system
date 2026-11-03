
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Step 4: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Display split information
print("\n--- DATASET SPLIT RESULTS ---")

print("Original dataset:", X.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
