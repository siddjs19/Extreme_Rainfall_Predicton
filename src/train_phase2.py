import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "phase2_model_comparison.csv"
)

# ============================================================
# LOAD MERGED DATA
# ============================================================

df = pd.read_csv(MERGED_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded dataset: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# TARGETS
# ============================================================

CLASSIFICATION_TARGET = "target_heavy_rain"
REGRESSION_TARGET = "target_rainfall_tomorrow"

# ============================================================
# IMD FEATURES ONLY
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

# ============================================================
# ERA5 ATMOSPHERIC FEATURES
# ============================================================

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

# Intentionally excluding:
# era5_total_precip_m
#
# Reason: it is precipitation information from the same day and
# can cause target leakage or make the comparison unfair.

combined_features = imd_features + era5_features

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    imd_features
    + era5_features
    + [CLASSIFICATION_TARGET, REGRESSION_TARGET]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nTrain period:")
print(train_df["date"].min().date(), "to", train_df["date"].max().date())

print("\nTest period:")
print(test_df["date"].min().date(), "to", test_df["date"].max().date())

print("\nTrain rows:", len(train_df))
print("Test rows:", len(test_df))

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate(feature_set, dataset_name):

    X_train = train_df[feature_set]
    X_test = test_df[feature_set]

    y_train_class = train_df[CLASSIFICATION_TARGET]
    y_test_class = test_df[CLASSIFICATION_TARGET]

    y_train_reg = train_df[REGRESSION_TARGET]
    y_test_reg = test_df[REGRESSION_TARGET]

    # --------------------------------------------------------
    # CLASSIFICATION MODEL
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

    classification_accuracy = accuracy_score(
        y_test_class,
        class_predictions
    )

    classification_f1 = f1_score(
        y_test_class,
        class_predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # REGRESSION MODEL
    # --------------------------------------------------------

    regressor = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    regressor.fit(X_train, y_train_reg)

    regression_predictions = regressor.predict(X_test)

    regression_mae = mean_absolute_error(
        y_test_reg,
        regression_predictions
    )

    regression_rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            regression_predictions
        )
    )

    regression_r2 = r2_score(
        y_test_reg,
        regression_predictions
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    return {
        "Dataset": dataset_name,
        "Number of Features": len(feature_set),
        "Classification Accuracy": classification_accuracy,
        "Classification F1 Score": classification_f1,
        "Regression MAE": regression_mae,
        "Regression RMSE": regression_rmse,
        "Regression R2 Score": regression_r2,
    }


# ============================================================
# TRAIN BOTH VERSIONS
# ============================================================

print("\nTraining IMD-only models...")

imd_results = train_and_evaluate(
    imd_features,
    "IMD Only"
)

print("IMD-only training complete.")

print("\nTraining IMD + ERA5 models...")

combined_results = train_and_evaluate(
    combined_features,
    "IMD + ERA5"
)

print("IMD + ERA5 training complete.")

# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame([
    imd_results,
    combined_results
])

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved comparison file:")
print(OUTPUT_FILE)

# ============================================================
# IMPROVEMENT CALCULATION
# ============================================================

imd = comparison_df.iloc[0]
combined = comparison_df.iloc[1]

print("\n" + "=" * 70)
print("IMPROVEMENT WITH ERA5 FEATURES")
print("=" * 70)

print(
    f"Accuracy improvement: "
    f"{combined['Classification Accuracy'] - imd['Classification Accuracy']:+.4f}"
)

print(
    f"F1 improvement: "
    f"{combined['Classification F1 Score'] - imd['Classification F1 Score']:+.4f}"
)

print(
    f"MAE change: "
    f"{combined['Regression MAE'] - imd['Regression MAE']:+.4f}"
)

print(
    f"RMSE change: "
    f"{combined['Regression RMSE'] - imd['Regression RMSE']:+.4f}"
)

print(
    f"R2 improvement: "
    f"{combined['Regression R2 Score'] - imd['Regression R2 Score']:+.4f}"
)