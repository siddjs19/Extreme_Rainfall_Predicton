import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
    "results",
    "phase3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(MERGED_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


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

classification_target = "target_heavy_rain"
regression_target = "target_rainfall_tomorrow"

required_columns = all_features + [
    classification_target,
    regression_target
]

df = df.dropna(subset=required_columns).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


X_train = train_df[all_features]
X_test = test_df[all_features]

y_train_class = train_df[classification_target]
y_train_reg = train_df[regression_target]


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

classifier.fit(X_train, y_train_class)


classification_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": classifier.feature_importances_
})

classification_importance = classification_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

classification_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================

regressor = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

regressor.fit(X_train, y_train_reg)


regression_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": regressor.feature_importances_
})

regression_importance = regression_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

regression_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES — HEAVY RAIN CLASSIFICATION")
print("=" * 70)

print(
    classification_importance.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("TOP FEATURES — NEXT-DAY RAINFALL REGRESSION")
print("=" * 70)

print(
    regression_importance.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PLOT CLASSIFICATION FEATURE IMPORTANCE
# ============================================================

top_classification = classification_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_classification["Feature"][::-1],
    top_classification["Importance"][::-1]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Feature")
plt.title("Top Features for Heavy Rain Classification")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "classification_feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT REGRESSION FEATURE IMPORTANCE
# ============================================================

top_regression = regression_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_regression["Feature"][::-1],
    top_regression["Importance"][::-1]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Feature")
plt.title("Top Features for Next-Day Rainfall Prediction")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "regression_feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GROUP IMPORTANCE: IMD VS ERA5
# ============================================================

def calculate_group_importance(importance_df):
    importance_df = importance_df.copy()

    importance_df["Source"] = importance_df["Feature"].apply(
        lambda x: "IMD" if x in imd_features else "ERA5"
    )

    return (
        importance_df.groupby("Source")["Importance"]
        .sum()
        .reset_index()
        .sort_values("Importance", ascending=False)
    )


classification_group_importance = calculate_group_importance(
    classification_importance
)

regression_group_importance = calculate_group_importance(
    regression_importance
)

classification_group_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_group_importance.csv"
    ),
    index=False
)

regression_group_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_group_importance.csv"
    ),
    index=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE BY DATA SOURCE")
print("=" * 70)

print("\nClassification:")
print(classification_group_importance.to_string(index=False))

print("\nRegression:")
print(regression_group_importance.to_string(index=False))


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_DIR)