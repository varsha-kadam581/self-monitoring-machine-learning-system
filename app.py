
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Self-Monitoring Machine Learning System")

st.write(
    "Monitor model performance, data drift "
    "and automated alerts."
)


# ==========================================
# 2. FILE PATHS
# ==========================================

REFERENCE_PATH = Path("data") / "reference_data.csv"
INCOMING_PATH = Path("data") / "incoming_data.csv"
TEST_PATH = Path("data") / "test_data.csv"
MODEL_PATH = Path("models") / "wine_quality_model.pkl"

THRESHOLD = 0.05
ACCURACY_THRESHOLD = 0.80


# ==========================================
# 3. LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_data():
    reference = pd.read_csv(REFERENCE_PATH)
    incoming = pd.read_csv(INCOMING_PATH)
    test = pd.read_csv(TEST_PATH)

    return reference, incoming, test


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    reference_data, default_incoming_data, test_data = load_data()
    model = load_model()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==========================================
# 4. INCOMING DATA UPLOAD
# ==========================================

st.sidebar.header("Upload New Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload incoming CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        incoming_data = pd.read_csv(uploaded_file)

        # Validate required feature columns
        required_columns = reference_data.columns.tolist()

        missing_columns = [
            column for column in required_columns
            if column not in incoming_data.columns
        ]

        if missing_columns:
            st.sidebar.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        # Keep only the 11 model input features
        incoming_data = incoming_data[required_columns]

        # Check for non-numeric values
        for column in required_columns:
            incoming_data[column] = pd.to_numeric(
                incoming_data[column],
                errors="coerce"
            )

        if incoming_data.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        if incoming_data.isnull().any().any():
            st.error(
                "Uploaded data contains missing or invalid values. "
                "Please correct the CSV and upload again."
            )
            st.stop()

        if not np.isfinite(incoming_data.to_numpy()).all():
            st.error(
                "Uploaded data contains infinite values."
            )
            st.stop()

        st.sidebar.success("CSV uploaded successfully!")

    except Exception as error:
        st.error(f"Unable to process uploaded CSV: {error}")
        st.stop()

else:
    incoming_data = default_incoming_data.copy()
    st.sidebar.info("Using default incoming dataset.")

# ==========================================
# 5. CHECK DATA COMPATIBILITY
# ==========================================

# Remove actual labels if present
incoming_data = incoming_data.drop(
    columns=["actual_quality"],
    errors="ignore"
)

# Get required feature columns
required_columns = reference_data.columns.tolist()

# Check for missing columns
missing_columns = [
    column for column in required_columns
    if column not in incoming_data.columns
]

if missing_columns:
    st.error(
        "Incoming data is missing columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Keep features in the correct order
incoming_data = incoming_data[required_columns]

# Check for empty data
if incoming_data.empty:
    st.error("Incoming dataset is empty.")
    st.stop()

# Check numeric values
for column in required_columns:
    incoming_data[column] = pd.to_numeric(
        incoming_data[column],
        errors="coerce"
    )

if incoming_data.isnull().any().any():
    st.error("Incoming data contains missing or invalid values.")
    st.stop()

if not np.isfinite(incoming_data.to_numpy()).all():
    st.error("Incoming data contains infinite values.")
    st.stop()

# ==========================================
# 6. MODEL PERFORMANCE MONITORING
# ==========================================

X_test = test_data.drop(columns=["actual_quality"])
y_actual = test_data["actual_quality"]

y_predicted = model.predict(X_test)

accuracy = accuracy_score(
    y_actual,
    y_predicted
)
# ==========================================
# INCOMING DATA PREDICTIONS
# ==========================================

incoming_predictions = model.predict(incoming_data)

prediction_results = incoming_data.copy()

prediction_results["Predicted Quality"] = incoming_predictions

print("Incoming predictions generated successfully!")
print("Total predictions:", len(incoming_predictions))


# ==========================================
# 7. DATA DRIFT DETECTION
# ==========================================

drift_results = []

for column in reference_data.columns:

    statistic, p_value = ks_2samp(
        reference_data[column],
        incoming_data[column]
    )

    status = (
        "DRIFT DETECTED"
        if p_value < THRESHOLD
        else "NO DRIFT"
    )

    drift_results.append({
        "Feature": column,
        "KS Statistic": statistic,
        "P-value": p_value,
        "Status": status
    })


drift_df = pd.DataFrame(drift_results)

drifted_features = drift_df[
    drift_df["Status"] == "DRIFT DETECTED"
]["Feature"].tolist()


# ==========================================
# 8. AUTOMATED ALERT SYSTEM
# ==========================================

alerts = []

if drifted_features:

    alerts.append(
        "Data drift detected in: "
        + ", ".join(drifted_features)
    )

if accuracy < ACCURACY_THRESHOLD:

    alerts.append(
        f"Model accuracy is low: {accuracy * 100:.2f}%"
    )


# ==========================================
# 9. SYSTEM OVERVIEW
# ==========================================

st.divider()

st.header("System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Total Features",
    len(reference_data.columns)
)

col3.metric(
    "Drifted Features",
    len(drifted_features)
)

col4.metric(
    "Incoming Records",
    len(incoming_data)
)


# ==========================================
# 10. AUTOMATED ALERTS
# ==========================================

st.divider()

st.header("Automated Alerts")

if alerts:

    for alert in alerts:
        st.warning(alert)

else:

    st.success(
        "System Healthy - No alerts detected."
    )


# ==========================================
# 11. DATA DRIFT MONITORING
# ==========================================

st.divider()

st.header("Data Drift Monitoring")

st.dataframe(
    drift_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 12. FEATURE DISTRIBUTION COMPARISON
# ==========================================

st.divider()

st.header("Feature Distribution Comparison")

selected_feature = st.selectbox(
    "Select a feature",
    reference_data.columns.tolist()
)

reference_values = reference_data[selected_feature]
incoming_values = incoming_data[selected_feature]

fig, ax = plt.subplots(figsize=(10, 5))

# Use common bin boundaries
all_values = np.concatenate([
    reference_values.to_numpy(),
    incoming_values.to_numpy()
])

bin_edges = np.histogram_bin_edges(
    all_values,
    bins=20
)

ax.hist(
    reference_values,
    bins=bin_edges,
    alpha=0.6,
    label="Reference Data"
)

ax.hist(
    incoming_values,
    bins=bin_edges,
    alpha=0.6,
    label="Incoming Data"
)

ax.set_title(
    f"Distribution Comparison: {selected_feature}"
)

ax.set_xlabel(selected_feature)
ax.set_ylabel("Number of Records")

ax.legend()
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)

plt.close(fig)

# ==========================================
# INCOMING DATA PREDICTIONS
# ==========================================

st.divider()
st.header("Incoming Data Predictions")

st.write(
    "Predicted wine quality categories for incoming records."
)

st.dataframe(
    prediction_results,
    width="stretch",
    height=450,
    hide_index=True
)

st.metric(
    "Total Predictions Generated",
    len(prediction_results)
)
# ==========================================
# PREDICTION CATEGORY SUMMARY
# ==========================================

st.divider()
st.header("Prediction Category Summary")

prediction_counts = (
    prediction_results["Predicted Quality"]
    .value_counts()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Low Quality",
    prediction_counts.get("Low", 0)
)

col2.metric(
    "Medium Quality",
    prediction_counts.get("Medium", 0)
)

col3.metric(
    "High Quality",
    prediction_counts.get("High", 0)
)
# ==========================================
# DOWNLOAD PREDICTION RESULTS
# ==========================================

st.divider()
st.header("Download Prediction Results")

csv_data = prediction_results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predictions as CSV",
    data=csv_data,
    file_name="wine_quality_predictions.csv",
    mime="text/csv"
)
# PREDICTION DISTRIBUTION CHART

st.divider()
st.header("Prediction Distribution")

fig, ax = plt.subplots(figsize=(3, 2), dpi=100)

ax.pie(
    prediction_counts,
    labels=prediction_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 6}
)

ax.set_title(
    "Wine Quality Prediction Distribution",
    fontsize=7
)

plt.tight_layout()

# Display chart at a controlled width
st.pyplot(fig, width=300)

plt.close(fig)
# ==========================================
# 13. MODEL PERFORMANCE DETAILS
# ==========================================

st.divider()

st.header("Model Performance Details")

report = classification_report(
    y_actual,
    y_predicted,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.success(
    "Monitoring dashboard loaded successfully."
)

st.caption(
    "Self-Monitoring Machine Learning System | "
    "Data Science Project"
)
