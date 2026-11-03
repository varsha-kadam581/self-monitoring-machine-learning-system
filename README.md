# Self-Monitoring Machine Learning System

## Project Overview

The **Self-Monitoring Machine Learning System** is a Data Science and Machine Learning project designed to monitor a trained machine learning model, detect changes in incoming data, evaluate model performance, and generate automated alerts.

The project uses a Random Forest Classifier to predict wine quality categories and a Streamlit dashboard to visualize monitoring results.

It helps demonstrate how machine learning models can be monitored after training instead of simply making predictions.

## Key Features

- **Data Preprocessing:** Loads and prepares the Wine Quality dataset.
- **Model Training:** Trains a Random Forest Classification model.
- **Model Performance Monitoring:** Evaluates model accuracy and classification metrics.
- **Data Drift Detection:** Uses the Kolmogorov-Smirnov (KS) statistical test to identify changes in incoming data.
- **Automated Alerts:** Generates warnings when data drift is detected or model accuracy falls below a defined threshold.
- **Interactive Dashboard:** Displays monitoring information through a Streamlit application.
- **CSV Upload:** Allows users to upload incoming data for monitoring.
- **Prediction System:** Predicts wine quality categories as Low, Medium, or High.
- **Prediction Visualization:** Displays prediction counts and a pie chart.
- **CSV Download:** Allows users to download prediction results.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit
- Joblib

## Dataset

This project uses the **Wine Quality - Red Wine Dataset** from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/186/wine+quality

Dataset details:
- Total records: 1,599
- Input features: 11
- Original target: Wine quality score
- Prediction categories: Low, Medium, High

## Machine Learning Algorithm

**Random Forest Classifier**

The model is trained to classify wine quality into three categories:

- Low: Quality score less than or equal to 4
- Medium: Quality score between 5 and 6
- High: Quality score greater than or equal to 7

The trained model is saved using Joblib.

## Data Drift Detection

The project uses the **Kolmogorov-Smirnov (KS) Test** to compare the feature distributions of reference data and incoming data.

Significance threshold: 0.05

- P-value less than 0.05: Drift detected.
- P-value greater than or equal to 0.05: No statistically significant drift detected.

The project includes simulated incoming data with a deliberately changed alcohol distribution to demonstrate drift detection.

## Model Performance

The trained Random Forest model achieved:

**Test Accuracy: 83.75%**

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The model has limitations, particularly when predicting the Low quality category, because the dataset contains relatively few Low quality samples.

## Project Structure

```text
ml-monitoring-system/
│
├── data/
│   ├── winequality-red.csv
│   ├── reference_data.csv
│   ├── test_data.csv
│   └── incoming_data.csv
│
├── models/
│   └── wine_quality_model.pkl
│
├── reports/
│
├── .venv/
│
├── inspect_data.py
├── prepare_data.py
├── split_data.py
├── train_model.py
├── detect_drift.py
├── simulate_data.py
├── monitor_performance.py
├── generate_alerts.py
├── app.py
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd ml-monitoring-system
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib streamlit scipy joblib
```

### 6. Train the machine learning model

```bash
python train_model.py
```

This trains the model and saves the reference and test datasets.

### 7. Generate simulated incoming data

```bash
python simulate_data.py
```

### 8. Run the monitoring dashboard

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

## How the System Works

1. Load and inspect the Wine Quality dataset.
2. Preprocess the dataset and categorize wine quality.
3. Split the dataset into training and testing data.
4. Train the Random Forest classification model.
5. Save the trained model and reference dataset.
6. Generate incoming data for monitoring.
7. Compare incoming data with reference data using the KS test.
8. Evaluate model performance on the original test dataset.
9. Generate automated alerts.
10. Display monitoring results and predictions through the Streamlit dashboard.

## Future Enhancements

The following features are planned for future development:

- Monitoring history with timestamps.
- Automatic model retraining.
- Model version comparison.
- Advanced monitoring reports.
- Improved alert notifications.

## Project Objective

The main objective of this project is to demonstrate how machine learning models can be monitored after deployment by identifying data distribution changes, evaluating model performance, and generating alerts through an interactive dashboard.

## Author

**Data Science and Machine Learning Project**

