import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MERGED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "merged_imd_era5_pune_features.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "results")

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "phase6_multi_event_forecasting_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "phase6_multi_event_forecasting_predictions.csv"
)

IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "phase6_multi_event_forecasting_feature_importance.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(MERGED_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Dataset shape:", df.shape)
print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)


# ============================================================
# CREATE TOMORROW'S MULTI-EVENT TARGET
# ============================================================

def classify_rainfall(rainfall):

    if rainfall == 0:
        return "Dry"

    elif rainfall <= 15:
        return "Light Rain"

    elif rainfall <= 64.5:
        return "Moderate Rain"

    elif rainfall <= 115.5:
        return "Heavy Rain"

    else:
        return "Extreme Rain"


# target_rainfall_tomorrow is already the next-day rainfall value
df["target_event_tomorrow"] = df[
    "target_rainfall_tomorrow"
].apply(classify_rainfall)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

imd_features = [
    "rainfall_mm",
    "year",
    "month",
    "day_of_year",
    "monsoon",
    "pre_monsoon",
    "post_monsoon",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_7",
    "rainfall_3day_sum",
    "rainfall_7day_sum",
    "rainfall_3day_mean",
    "rainfall_7day_mean",
    "rainfall_7day_max",
    "rain_today_flag",
    "rainy_days_last_7",
]

era5_features = [
    "pune_temp_mean_c",
    "pune_temp_max_c",
    "pune_temp_min_c",
    "pune_dewpoint_mean_c",
    "pune_pressure_mean_pa",
    "pune_pressure_min_pa",
    "pune_wind_mean_ms",
    "pune_wind_max_ms",
    "pune_rh_mean_pct",
    "pune_dewpoint_depression_mean_c",
]

all_features = imd_features + era5_features

TARGET = "target_event_tomorrow"

required_columns = all_features + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


# ============================================================
# EVENT DISTRIBUTION
# ============================================================

print("\nTomorrow's event distribution:")

print(
    df[TARGET]
    .value_counts()
)

print(
    "\nTomorrow's event percentages:"
)

print(
    (df[TARGET].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[all_features]
X_test = test_df[all_features]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

print("\nTrain rows:", len(train_df))
print("Test rows:", len(test_df))

print(
    "Train period:",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Test period:",
    test_df["date"].min().date(),
    "to",
    test_df["date"].max().date()
)


# ============================================================
# TRAIN MULTI-EVENT FORECASTING MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

print("\nTraining leakage-free multi-event forecasting model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Training complete.")


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

event_labels = [
    "Dry",
    "Light Rain",
    "Moderate Rain",
    "Heavy Rain",
    "Extreme Rain"
]

report_dict = classification_report(
    y_test,
    predictions,
    labels=event_labels,
    output_dict=True,
    zero_division=0
)

print("\n" + "=" * 80)
print("PHASE 6 — LEAKAGE-FREE MULTI-EVENT FORECASTING")
print("=" * 80)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=event_labels,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=event_labels
)

cm_df = pd.DataFrame(
    cm,
    index=event_labels,
    columns=event_labels
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "date",
        "target_rainfall_tomorrow"
    ]
].copy()

prediction_df["actual_event_tomorrow"] = y_test.values
prediction_df["predicted_event_tomorrow"] = predictions

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY RESULTS
# ============================================================

results_df = pd.DataFrame([
    {
        "Model": "IMD + ERA5 Leakage-Free Multi-Event Forecasting",
        "Number of Features": len(all_features),
        "Number of Classes": len(event_labels),
        "Accuracy": accuracy,
        "Macro F1 Score": macro_f1,
        "Weighted F1 Score": weighted_f1
    }
])

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 80)
print("TOP FEATURES FOR TOMORROW'S EVENT PREDICTION")
print("=" * 80)

print(
    importance_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE CLASS-WISE METRICS
# ============================================================

class_metrics = []

for event in event_labels:

    if event in report_dict:

        class_metrics.append({
            "Event": event,
            "Precision": report_dict[event]["precision"],
            "Recall": report_dict[event]["recall"],
            "F1 Score": report_dict[event]["f1-score"],
            "Support": report_dict[event]["support"]
        })

class_metrics_df = pd.DataFrame(class_metrics)

class_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_multi_event_class_metrics.csv"
    ),
    index=False
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(RESULTS_FILE)
print(PREDICTIONS_FILE)
print(IMPORTANCE_FILE)
print(
    os.path.join(
        OUTPUT_DIR,
        "phase6_multi_event_class_metrics.csv"
    )
)