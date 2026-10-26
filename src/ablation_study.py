import os
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase4_ablation_results.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(MERGED_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# TARGETS
# ============================================================

CLASSIFICATION_TARGET = "target_heavy_rain"
REGRESSION_TARGET = "target_rainfall_tomorrow"


# ============================================================
# FEATURE GROUPS
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

temperature_features = [
    "pune_temp_mean_c",
    "pune_temp_max_c",
    "pune_temp_min_c",
]

moisture_features = [
    "pune_dewpoint_mean_c",
    "pune_rh_mean_pct",
    "pune_dewpoint_depression_mean_c",
]

pressure_features = [
    "pune_pressure_mean_pa",
    "pune_pressure_min_pa",
]

wind_features = [
    "pune_wind_mean_ms",
    "pune_wind_max_ms",
]

all_era5_features = (
    temperature_features
    + moisture_features
    + pressure_features
    + wind_features
)

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {
    "IMD Only": imd_features,

    "IMD + Temperature": (
        imd_features
        + temperature_features
    ),

    "IMD + Moisture": (
        imd_features
        + moisture_features
    ),

    "IMD + Pressure": (
        imd_features
        + pressure_features
    ),

    "IMD + Wind": (
        imd_features
        + wind_features
    ),

    "IMD + All ERA5": (
        imd_features
        + all_era5_features
    ),
}


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = (
    imd_features
    + all_era5_features
    + [
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET
    ]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("Train rows:", len(train_df))
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
# MODEL TRAINING FUNCTION
# ============================================================

def evaluate_experiment(experiment_name, features):

    X_train = train_df[features]
    X_test = test_df[features]

    y_train_class = train_df[CLASSIFICATION_TARGET]
    y_test_class = test_df[CLASSIFICATION_TARGET]

    y_train_reg = train_df[REGRESSION_TARGET]
    y_test_reg = test_df[REGRESSION_TARGET]

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )

    classifier.fit(X_train, y_train_class)

    class_predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test_class,
        class_predictions
    )

    f1 = f1_score(
        y_test_class,
        class_predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regressor = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    regressor.fit(X_train, y_train_reg)

    reg_predictions = regressor.predict(X_test)

    mae = mean_absolute_error(
        y_test_reg,
        reg_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            reg_predictions
        )
    )

    r2 = r2_score(
        y_test_reg,
        reg_predictions
    )

    return {
        "Experiment": experiment_name,
        "Number of Features": len(features),
        "Accuracy": accuracy,
        "F1 Score": f1,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []

for experiment_name, features in experiments.items():

    print("\n" + "-" * 60)
    print(f"Running: {experiment_name}")
    print(f"Features: {len(features)}")

    result = evaluate_experiment(
        experiment_name,
        features
    )

    results.append(result)

    print(
        f"Accuracy: {result['Accuracy']:.4f}"
    )

    print(
        f"F1 Score: {result['F1 Score']:.4f}"
    )

    print(
        f"MAE: {result['MAE']:.4f}"
    )

    print(
        f"RMSE: {result['RMSE']:.4f}"
    )

    print(
        f"R2 Score: {result['R2 Score']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CALCULATE IMPROVEMENT OVER IMD BASELINE
# ============================================================

baseline = results_df.iloc[0]

results_df["Accuracy Improvement"] = (
    results_df["Accuracy"]
    - baseline["Accuracy"]
)

results_df["F1 Improvement"] = (
    results_df["F1 Score"]
    - baseline["F1 Score"]
)

results_df["MAE Improvement"] = (
    baseline["MAE"]
    - results_df["MAE"]
)

results_df["RMSE Improvement"] = (
    baseline["RMSE"]
    - results_df["RMSE"]
)

results_df["R2 Improvement"] = (
    results_df["R2 Score"]
    - baseline["R2 Score"]
)


# Save updated results
results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n" + "=" * 90)
print("PHASE 4 — ABLATION STUDY RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved to:")
print(OUTPUT_FILE)


# ============================================================
# BEST EXPERIMENTS
# ============================================================

best_classification = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_regression = results_df.loc[
    results_df["RMSE"].idxmin()
]

print("\n" + "=" * 90)
print("BEST CLASSIFICATION EXPERIMENT")
print("=" * 90)

print(
    best_classification[
        [
            "Experiment",
            "Accuracy",
            "F1 Score",
            "Accuracy Improvement",
            "F1 Improvement"
        ]
    ].to_string()
)


print("\n" + "=" * 90)
print("BEST REGRESSION EXPERIMENT")
print("=" * 90)

print(
    best_regression[
        [
            "Experiment",
            "MAE",
            "RMSE",
            "R2 Score",
            "MAE Improvement",
            "RMSE Improvement",
            "R2 Improvement"
        ]
    ].to_string()
)