# ============================================================
# WEATHERGPT RESEARCH - PHASE 1
# REGIONAL RAINFALL MODEL COMPARISON
#
# Region: Konkan / Western Maharashtra
# Data: IMD RF25 rainfall NetCDF files, 2021-2025
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. Gradient Boosting
#   4. XGBoost
#
# Tasks:
#   A. Predict next-day rainfall amount
#   B. Predict next-day heavy rainfall occurrence
#
# Outputs:
#   data/regional_rainfall_features.csv
#   results/model_comparison.csv
#   results/feature_importance_xgboost.csv
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

from sklearn.feature_selection import mutual_info_classif

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Project root is one level above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_DATA = os.path.join(
    DATA_DIR,
    "regional_rainfall_features.csv"
)

OUTPUT_RESULTS = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

OUTPUT_FEATURES = os.path.join(
    RESULTS_DIR,
    "feature_importance_xgboost.csv"
)


# ------------------------------------------------------------
# REGION CONFIGURATION
# ------------------------------------------------------------

# Konkan / Western Maharashtra region
#
# Includes approximately:
# Mumbai
# Pune
# Raigad
# Ratnagiri
# Sindhudurg
# Parts of Western Maharashtra
#
# Adjust later if necessary.

MIN_LAT = 15.0
MAX_LAT = 21.0

MIN_LON = 72.0
MAX_LON = 75.5


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42

# Heavy rainfall threshold for regional daily rainfall.
#
# We use 25 mm initially because this is a regional rainfall
# experiment and should contain significantly more events than
# the India-wide spatial average.
#
# Later we can compare multiple thresholds.
HEAVY_RAIN_THRESHOLD = 25.0

# Number of previous days used as features
LAG_DAYS = [1, 2, 3, 7]


# ============================================================
# 3. FIND NETCDF FILES
# ============================================================

print("=" * 70)
print("WEATHERGPT PHASE 1 - REGIONAL RAINFALL EXPERIMENT")
print("=" * 70)

print("\nSearching for NetCDF files...")

nc_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.nc"))
)

if len(nc_files) == 0:
    raise FileNotFoundError(
        f"No .nc files found in: {DATA_DIR}\n"
        "Make sure your NetCDF files are inside the data folder."
    )

print(f"\nFound {len(nc_files)} NetCDF files:")

for file in nc_files:
    print(" -", os.path.basename(file))


# ============================================================
# 4. INSPECT DATASET STRUCTURE
# ============================================================

print("\nInspecting first dataset...")

sample_ds = xr.open_dataset(
    nc_files[0],
    engine="netcdf4"
)

print(sample_ds)

print("\nCoordinates:")
print(list(sample_ds.coords))

print("\nVariables:")
print(list(sample_ds.data_vars))


# ============================================================
# 5. IDENTIFY RAINFALL VARIABLE AND DIMENSIONS
# ============================================================

# IMD RF25 files generally contain rainfall variable RF25,
# but we detect it automatically where possible.

possible_rainfall_names = [
    "RAINFALL",
    "rainfall",
    "RF25",
    "rf25",
    "RAINFALL_MM",
    "precipitation",
    "precip"
]

rainfall_var = None

for name in possible_rainfall_names:
    if name in sample_ds.data_vars:
        rainfall_var = name
        break

if rainfall_var is None:

    # Fallback: find the first variable containing rain-related text
    for name in sample_ds.data_vars:
        if any(
            word in name.lower()
            for word in ["rain", "precip", "rf"]
        ):
            rainfall_var = name
            break

if rainfall_var is None:
    raise ValueError(
        "Could not automatically identify rainfall variable.\n"
        f"Available variables: {list(sample_ds.data_vars)}"
    )

print("\nSelected rainfall variable:", rainfall_var)


# Identify latitude and longitude names

def find_coordinate(ds, possible_names):
    for name in possible_names:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name
    return None


lat_name = find_coordinate(
    sample_ds,
    ["LATITUDE", "latitude", "lat", "Latitude"]
)

lon_name = find_coordinate(
    sample_ds,
    ["LONGITUDE", "longitude", "lon", "Longitude"]
)

time_name = find_coordinate(
    sample_ds,
    ["TIME", "time", "Time", "date", "DATE"]
)

if lat_name is None or lon_name is None or time_name is None:
    raise ValueError(
        "Could not identify latitude, longitude, or time dimensions.\n"
        f"Coordinates: {list(sample_ds.coords)}"
    )

print("Latitude coordinate:", lat_name)
print("Longitude coordinate:", lon_name)
print("Time coordinate:", time_name)


sample_ds.close()


# ============================================================
# 6. PROCESS EACH YEAR WITHOUT CONCATENATING HUGE ARRAYS
# ============================================================

print("\nProcessing regional rainfall data...")

regional_daily_datasets = []

for file in nc_files:

    print("\nOpening:", os.path.basename(file))

    ds = xr.open_dataset(
        file,
        engine="netcdf4"
    )

    # --------------------------------------------------------
    # Select the regional geographic area
    # --------------------------------------------------------

    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    # Determine whether latitude/longitude are ascending
    lat_min_actual = min(MIN_LAT, MAX_LAT)
    lat_max_actual = max(MIN_LAT, MAX_LAT)

    lon_min_actual = min(MIN_LON, MAX_LON)
    lon_max_actual = max(MIN_LON, MAX_LON)

    # Select using boolean masks.
    # This works regardless of coordinate ordering.

    lat_mask = (
        (ds[lat_name] >= lat_min_actual) &
        (ds[lat_name] <= lat_max_actual)
    )

    lon_mask = (
        (ds[lon_name] >= lon_min_actual) &
        (ds[lon_name] <= lon_max_actual)
    )

    regional_ds = ds.where(
        lat_mask & lon_mask,
        drop=True
    )

    if regional_ds.sizes.get(lat_name, 0) == 0:
        raise ValueError(
            f"No latitude points found in selected region for {file}"
        )

    if regional_ds.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            f"No longitude points found in selected region for {file}"
        )

    print(
        f"Regional grid: "
        f"{regional_ds.sizes[lat_name]} latitude × "
        f"{regional_ds.sizes[lon_name]} longitude"
    )

    # --------------------------------------------------------
    # Extract rainfall
    # --------------------------------------------------------

    rainfall = regional_ds[rainfall_var]

    # --------------------------------------------------------
    # Calculate daily regional rainfall
    #
    # IMPORTANT:
    # We use MAXIMUM grid-cell rainfall, not spatial mean.
    #
    # This preserves heavy rainfall occurring in any part of
    # the selected region.
    # --------------------------------------------------------

    daily_regional_rainfall = rainfall.max(
        dim=[lat_name, lon_name],
        skipna=True
    )

    # Convert to DataFrame immediately.
    # This avoids creating the massive full spatial DataFrame.

    temp_df = daily_regional_rainfall.to_dataframe(
        name="rainfall_mm"
    ).reset_index()

    temp_df["time"] = pd.to_datetime(
        temp_df[time_name]
    )

    temp_df = temp_df[
        ["time", "rainfall_mm"]
    ]

    temp_df = temp_df.dropna(
        subset=["rainfall_mm"]
    )

    temp_df = temp_df.sort_values(
        "time"
    )

    temp_df = temp_df.drop_duplicates(
        subset=["time"]
    )

    regional_daily_datasets.append(
        temp_df
    )

    ds.close()


# ============================================================
# 7. COMBINE ALL YEARS
# ============================================================

print("\nCombining years...")

df = pd.concat(
    regional_daily_datasets,
    ignore_index=True
)

df = df.sort_values(
    "time"
).reset_index(drop=True)

df = df.drop_duplicates(
    subset=["time"]
)

print("\nCombined dataset shape:", df.shape)

print("\nDate range:")
print("Start:", df["time"].min())
print("End:  ", df["time"].max())

print("\nRainfall statistics:")
print(df["rainfall_mm"].describe())


# ============================================================
# 8. CHECK REGIONAL RAINFALL DISTRIBUTION
# ============================================================

print("\nRainfall distribution:")

thresholds = [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    100
]

for threshold in thresholds:
    count = (
        df["rainfall_mm"] >= threshold
    ).sum()

    print(
        f"Days >= {threshold:>3} mm: {count}"
    )

print(
    "\nMaximum regional rainfall:",
    df["rainfall_mm"].max(),
    "mm"
)


# ============================================================
# 9. CREATE WEATHER FEATURES
# ============================================================

print("\nCreating features...")

# ------------------------------------------------------------
# Temporal features
# ------------------------------------------------------------

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear

df["monsoon"] = df["month"].isin(
    [6, 7, 8, 9]
).astype(int)

df["pre_monsoon"] = df["month"].isin(
    [3, 4, 5]
).astype(int)

df["post_monsoon"] = df["month"].isin(
    [10, 11]
).astype(int)


# ------------------------------------------------------------
# Rainfall lag features
# ------------------------------------------------------------

for lag in LAG_DAYS:

    df[f"rainfall_lag_{lag}"] = (
        df["rainfall_mm"].shift(lag)
    )


# ------------------------------------------------------------
# Rolling rainfall accumulation
# ------------------------------------------------------------

df["rainfall_3day_sum"] = (
    df["rainfall_mm"]
    .shift(1)
    .rolling(window=3)
    .sum()
)

df["rainfall_7day_sum"] = (
    df["rainfall_mm"]
    .shift(1)
    .rolling(window=7)
    .sum()
)

df["rainfall_3day_mean"] = (
    df["rainfall_mm"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

df["rainfall_7day_mean"] = (
    df["rainfall_mm"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["rainfall_7day_max"] = (
    df["rainfall_mm"]
    .shift(1)
    .rolling(window=7)
    .max()
)


# ------------------------------------------------------------
# Rainfall occurrence history
# ------------------------------------------------------------

df["rain_today_flag"] = (
    df["rainfall_mm"] > 0
).astype(int)

df["rainy_days_last_7"] = (
    df["rain_today_flag"]
    .shift(1)
    .rolling(window=7)
    .sum()
)


# ============================================================
# 10. CREATE TARGETS
# ============================================================

print("\nCreating prediction targets...")

# Tomorrow's rainfall amount
df["target_rainfall_tomorrow"] = (
    df["rainfall_mm"].shift(-1)
)

# Tomorrow's heavy rainfall occurrence
df["target_heavy_rain"] = np.where(
    df["target_rainfall_tomorrow"].isna(),
    np.nan,
    (
        df["target_rainfall_tomorrow"]
        >= HEAVY_RAIN_THRESHOLD
    ).astype(int)
)


# Remove rows that do not have enough historical data
# or do not have tomorrow's rainfall target.

df = df.dropna(
    subset=[
        "target_rainfall_tomorrow",
        "target_heavy_rain"
    ]
).reset_index(drop=True)


df["target_heavy_rain"] = (
    df["target_heavy_rain"].astype(int)
)


# ============================================================
# 11. DEFINE FEATURES
# ============================================================

FEATURES = [
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

    "rainy_days_last_7"
]


df = df.dropna(
    subset=FEATURES
).reset_index(drop=True)


print("\nFinal dataset shape:", df.shape)

print("\nTarget distribution:")

print(
    df["target_heavy_rain"]
    .value_counts()
)

print("\nHeavy rainfall percentage:")

print(
    df["target_heavy_rain"].mean() * 100,
    "%"
)


# ============================================================
# 12. SAVE PROCESSED DATASET
# ============================================================

print("\nSaving processed regional dataset...")

df.to_csv(
    OUTPUT_DATA,
    index=False
)

print(
    "Saved:",
    OUTPUT_DATA
)


# ============================================================
# 13. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nSplitting dataset chronologically...")

# IMPORTANT:
# We do NOT randomly shuffle time-series data.
#
# Training:      2021-2023
# Validation:     2024
# Testing:        2025

train_df = df[
    df["time"].dt.year <= 2023
].copy()

val_df = df[
    df["time"].dt.year == 2024
].copy()

test_df = df[
    df["time"].dt.year == 2025
].copy()

print("\nTrain:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ============================================================
# 14. PREPARE FEATURES AND TARGETS
# ============================================================

X_train = train_df[FEATURES]
X_val = val_df[FEATURES]
X_test = test_df[FEATURES]

# Rainfall regression targets

y_train_reg = train_df[
    "target_rainfall_tomorrow"
]

y_val_reg = val_df[
    "target_rainfall_tomorrow"
]

y_test_reg = test_df[
    "target_rainfall_tomorrow"
]

# Heavy rainfall classification targets

y_train_cls = train_df[
    "target_heavy_rain"
]

y_val_cls = val_df[
    "target_heavy_rain"
]

y_test_cls = test_df[
    "target_heavy_rain"
]


# ============================================================
# 15. CHECK TARGET BALANCE
# ============================================================

print("\nTarget balance:")

print(
    "\nTRAIN"
)

print(y_train_cls.value_counts())

print(
    "\nVALIDATION"
)

print(y_val_cls.value_counts())

print(
    "\nTEST"
)

print(y_test_cls.value_counts())


if y_train_cls.nunique() < 2:
    raise ValueError(
        "\nTraining data contains only one rainfall class.\n"
        "Lower HEAVY_RAIN_THRESHOLD or choose a heavier-rainfall region."
    )


# ============================================================
# 16. TRAIN CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING HEAVY RAINFALL CLASSIFICATION MODELS")
print("=" * 70)


classification_models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


classification_results = []


for model_name, model in classification_models.items():

    print(f"\nTraining {model_name}...")

    model.fit(
        X_train,
        y_train_cls
    )

    # Validation predictions

    val_pred = model.predict(X_val)

    val_prob = model.predict_proba(X_val)[:, 1]

    # Test predictions

    test_pred = model.predict(X_test)

    test_prob = model.predict_proba(X_test)[:, 1]

    # Metrics

    val_f1 = f1_score(
        y_val_cls,
        val_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test_cls,
        test_pred,
        zero_division=0
    )

    val_precision = precision_score(
        y_val_cls,
        val_pred,
        zero_division=0
    )

    test_precision = precision_score(
        y_test_cls,
        test_pred,
        zero_division=0
    )

    val_recall = recall_score(
        y_val_cls,
        val_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test_cls,
        test_pred,
        zero_division=0
    )

    try:
        val_roc_auc = roc_auc_score(
            y_val_cls,
            val_prob
        )
    except ValueError:
        val_roc_auc = np.nan

    try:
        test_roc_auc = roc_auc_score(
            y_test_cls,
            test_prob
        )
    except ValueError:
        test_roc_auc = np.nan

    try:
        val_pr_auc = average_precision_score(
            y_val_cls,
            val_prob
        )
    except ValueError:
        val_pr_auc = np.nan

    try:
        test_pr_auc = average_precision_score(
            y_test_cls,
            test_prob
        )
    except ValueError:
        test_pr_auc = np.nan

    classification_results.append({

        "Task": "Heavy Rainfall Classification",

        "Model": model_name,

        "Validation_F1": val_f1,
        "Test_F1": test_f1,

        "Validation_Precision": val_precision,
        "Test_Precision": test_precision,

        "Validation_Recall": val_recall,
        "Test_Recall": test_recall,

        "Validation_ROC_AUC": val_roc_auc,
        "Test_ROC_AUC": test_roc_auc,

        "Validation_PR_AUC": val_pr_auc,
        "Test_PR_AUC": test_pr_auc

    })

    print(
        f"Test F1: {test_f1:.4f} | "
        f"Test Precision: {test_precision:.4f} | "
        f"Test Recall: {test_recall:.4f}"
    )


# ============================================================
# 17. TRAIN REGRESSION MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RAINFALL REGRESSION MODELS")
print("=" * 70)


regression_models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


regression_results = []


for model_name, model in regression_models.items():

    print(f"\nTraining {model_name}...")

    model.fit(
        X_train,
        y_train_reg
    )

    val_pred = model.predict(X_val)

    test_pred = model.predict(X_test)

    val_mae = mean_absolute_error(
        y_val_reg,
        val_pred
    )

    test_mae = mean_absolute_error(
        y_test_reg,
        test_pred
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val_reg,
            val_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            test_pred
        )
    )

    val_r2 = r2_score(
        y_val_reg,
        val_pred
    )

    test_r2 = r2_score(
        y_test_reg,
        test_pred
    )

    regression_results.append({

        "Task": "Rainfall Amount Regression",

        "Model": model_name,

        "Validation_MAE": val_mae,
        "Test_MAE": test_mae,

        "Validation_RMSE": val_rmse,
        "Test_RMSE": test_rmse,

        "Validation_R2": val_r2,
        "Test_R2": test_r2

    })

    print(
        f"Test MAE: {test_mae:.4f} mm | "
        f"Test RMSE: {test_rmse:.4f} mm | "
        f"Test R²: {test_r2:.4f}"
    )


# ============================================================
# 18. COMBINE AND SAVE MODEL RESULTS
# ============================================================

print("\nSaving model comparison results...")

results_df = pd.DataFrame(
    classification_results + regression_results
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_RESULTS
)

print("\nMODEL COMPARISON RESULTS")
print(results_df.to_string(index=False))


# ============================================================
# 19. FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


# Train final XGBoost classifier for feature importance

final_xgb_classifier = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_xgb_classifier.fit(
    X_train,
    y_train_cls
)


importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": final_xgb_classifier.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


importance_df.to_csv(
    OUTPUT_FEATURES,
    index=False
)

print("\nXGBoost Feature Importance:")

print(importance_df.to_string(index=False))


# ============================================================
# 20. MUTUAL INFORMATION ANALYSIS
# ============================================================

print("\nCalculating Mutual Information...")

mi_scores = mutual_info_classif(
    X_train,
    y_train_cls,
    random_state=RANDOM_STATE
)

mi_df = pd.DataFrame({

    "Feature": FEATURES,

    "Mutual_Information": mi_scores

}).sort_values(
    "Mutual_Information",
    ascending=False
).reset_index(drop=True)

mi_output = os.path.join(
    RESULTS_DIR,
    "mutual_information.csv"
)

mi_df.to_csv(
    mi_output,
    index=False
)

print("\nMutual Information ranking:")

print(mi_df.to_string(index=False))


# ============================================================
# 21. PLOT RAINFALL DISTRIBUTION
# ============================================================

print("\nGenerating plots...")

plt.figure(figsize=(12, 5))

plt.plot(
    df["time"],
    df["rainfall_mm"],
    linewidth=0.8
)

plt.axhline(
    HEAVY_RAIN_THRESHOLD,
    linestyle="--",
    label=f"Heavy Rain Threshold ({HEAVY_RAIN_THRESHOLD} mm)"
)

plt.title(
    "Regional Daily Rainfall - Konkan / Western Maharashtra"
)

plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")

plt.legend()

plt.tight_layout()

rainfall_plot_path = os.path.join(
    RESULTS_DIR,
    "regional_rainfall_distribution.png"
)

plt.savefig(
    rainfall_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# 22. PLOT FEATURE IMPORTANCE
# ============================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top XGBoost Features for Heavy Rainfall Prediction"
)

plt.xlabel("Feature Importance")

plt.tight_layout()

importance_plot_path = os.path.join(
    RESULTS_DIR,
    "xgboost_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 COMPLETED")
print("=" * 70)

print("\nFiles generated:")

print("1.", OUTPUT_DATA)
print("2.", OUTPUT_RESULTS)
print("3.", OUTPUT_FEATURES)
print("4.", mi_output)
print("5.", rainfall_plot_path)
print("6.", importance_plot_path)

print("\nFinal dataset:")
print(df.shape)

print("\nDate range:")
print(df["time"].min(), "to", df["time"].max())

print("\nHeavy rainfall threshold:")
print(HEAVY_RAIN_THRESHOLD, "mm")

print("\nHeavy rainfall events:")
print(df["target_heavy_rain"].sum())

print("\nThe next research phase can compare:")
print("- Individual models")
print("- Feature importance")
print("- Ensemble models")
print("- Multitask learning")