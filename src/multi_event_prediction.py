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
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_multi_event_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_multi_event_predictions.csv"
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
# CREATE MULTI-EVENT TARGET
# ============================================================
#
# Thresholds are based on daily rainfall categories:
#
# 0 mm              -> Dry
# 0–15 mm           -> Light Rain
# 15–64.5 mm        -> Moderate Rain
# 64.5–115.5 mm     -> Heavy Rain
# >115.5 mm         -> Extreme Rain
#
# These thresholds are commonly used for Indian daily rainfall
# intensity classification.
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


df["rainfall_event"] = df["rainfall_mm"].apply(
    classify_rainfall
)

print("\nEvent distribution:")
print(
    df["rainfall_event"]
    .value_counts()
)


# ============================================================
# FEATURES
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

required_columns = all_features + ["rainfall_event"]

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[all_features]
X_test = test_df[all_features]

y_train = train_df["rainfall_event"]
y_test = test_df["rainfall_event"]

print("\nTrain rows:", len(train_df))
print("Test rows:", len(test_df))


# ============================================================
# MULTI-CLASS CLASSIFIER
# ============================================================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

print("\nTraining multi-event classifier...")

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

print("\n" + "=" * 70)
print("PHASE 5 — MULTI-EVENT CLASSIFICATION RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

event_labels = [
    "Dry",
    "Light Rain",
    "Moderate Rain",
    "Heavy Rain",
    "Extreme Rain"
]

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
    ["date", "rainfall_mm", "rainfall_event"]
].copy()

prediction_df["predicted_event"] = predictions

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY RESULTS
# ============================================================

results_df = pd.DataFrame([
    {
        "Model": "IMD + ERA5 Multi-Event Classifier",
        "Number of Features": len(all_features),
        "Number of Classes": len(event_labels),
        "Accuracy": accuracy,
        "Macro F1 Score": macro_f1,
        "Weighted F1 Score": weighted_f1
    }
])

results_df.to_csv(
    OUTPUT_FILE,
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
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_multi_event_feature_importance.csv"
    ),
    index=False
)


print("\nFiles saved:")
print(OUTPUT_FILE)
print(PREDICTIONS_FILE)
print(
    os.path.join(
        OUTPUT_DIR,
        "phase5_multi_event_feature_importance.csv"
    )
)