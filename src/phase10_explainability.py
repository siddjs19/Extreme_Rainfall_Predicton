"""
Phase 10: Model Explainability
WeatherGPT Research Project

Generates:
1. Regression permutation importance
2. Classification permutation importance
3. Feature-importance plots
4. Optional SHAP summaries
5. Heavy/Extreme rainfall error explanations
6. Summary metrics

Dataset:
data/merged_imd_era5_pune_features.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\research\WeatherGPT-Research")
DATA_PATH = BASE_DIR / "data" / "merged_imd_era5_pune_features.csv"
RESULT_DIR = BASE_DIR / "results" / "phase10_explainability"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns that should not be used as predictors
EXCLUDE_COLUMNS = [
    "date",
    "target_rainfall_tomorrow",
    "target_heavy_rain",
    "actual_event_tomorrow",
    "predicted_event_tomorrow",
    "actual_rainfall_tomorrow",
    "predicted_rainfall_tomorrow",
]


# ============================================================
# HELPERS
# ============================================================

def classify_rainfall(value):
    """
    IMD-style next-day rainfall categories used in the project.
    """
    if value <= 0:
        return "Dry"
    elif value <= 15:
        return "Light Rain"
    elif value <= 64.5:
        return "Moderate Rain"
    elif value <= 115.5:
        return "Heavy Rain"
    else:
        return "Extreme Rain"


def save_bar_plot(importance_df, title, output_path, top_n=15):
    """
    Save horizontal permutation-importance plot.
    """
    plot_df = importance_df.head(top_n).sort_values("importance_mean")

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )
    plt.xlabel("Mean decrease in model performance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_permutation_importance(
    model,
    X_test,
    y_test,
    scoring,
    model_name,
):
    """
    Calculate permutation importance and save CSV + PNG.
    """

    print(f"\nCalculating permutation importance: {model_name}")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df["importance_mean_abs"] = (
        importance_df["importance_mean"].abs()
    )

    importance_df = importance_df.sort_values(
        "importance_mean_abs",
        ascending=False,
    ).reset_index(drop=True)

    csv_name = (
        "regression_permutation_importance.csv"
        if model_name == "regression"
        else "classification_permutation_importance.csv"
    )

    png_name = (
        "regression_feature_importance.png"
        if model_name == "regression"
        else "classification_feature_importance.png"
    )

    importance_df.to_csv(RESULT_DIR / csv_name, index=False)

    save_bar_plot(
        importance_df,
        f"{model_name.title()} Permutation Feature Importance",
        RESULT_DIR / png_name,
    )

    print("\nTop features:")
    print(
        importance_df[
            ["feature", "importance_mean", "importance_std"]
        ].head(15).to_string(index=False)
    )

    return importance_df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 10: MODEL EXPLAINABILITY")
print("=" * 70)

print(f"\nLoading dataset:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# CREATE TARGETS
# ============================================================

if "target_rainfall_tomorrow" not in df.columns:
    raise ValueError(
        "Missing target_rainfall_tomorrow column in dataset."
    )

df["event_target"] = df["target_rainfall_tomorrow"].apply(
    classify_rainfall
)


# ============================================================
# PREPARE FEATURES
# ============================================================

exclude = set(EXCLUDE_COLUMNS)
exclude.add("event_target")

feature_columns = [
    column
    for column in df.columns
    if column not in exclude
]

X = df[feature_columns].copy()

# Convert categorical/object/string columns to numeric
for column in X.columns:
    if (
        pd.api.types.is_object_dtype(X[column])
        or pd.api.types.is_string_dtype(X[column])
    ):
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

# Convert datetime columns safely
for column in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[column]):
        X[column] = X[column].astype("int64") / 10**9

# Replace infinities and fill missing values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

# Ensure numeric features only
X = X.select_dtypes(include=[np.number])

y_reg = pd.to_numeric(
    df["target_rainfall_tomorrow"],
    errors="coerce",
)

valid_rows = y_reg.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y_reg = y_reg.loc[valid_rows].reset_index(drop=True)
event_target = df.loc[valid_rows, "event_target"].reset_index(drop=True)

print(f"\nNumber of features: {X.shape[1]}")
print(f"Features: {list(X.columns)}")


# ============================================================
# TIME-ORDERED TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(X) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_reg_train = y_reg.iloc[:split_index].copy()
y_reg_test = y_reg.iloc[split_index:].copy()

event_train = event_target.iloc[:split_index].copy()
event_test = event_target.iloc[split_index:].copy()

print("\nTrain/Test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTest event distribution:")
print(event_test.value_counts())


# ============================================================
# REGRESSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL")
print("=" * 70)

regressor = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

regressor.fit(X_train, y_reg_train)

reg_predictions = regressor.predict(X_test)

reg_mae = mean_absolute_error(y_reg_test, reg_predictions)
reg_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_predictions))
reg_r2 = r2_score(y_reg_test, reg_predictions)

print(f"\nMAE:  {reg_mae:.4f}")
print(f"RMSE: {reg_rmse:.4f}")
print(f"R²:   {reg_r2:.4f}")

reg_importance = calculate_permutation_importance(
    regressor,
    X_test,
    y_reg_test,
    scoring="neg_mean_absolute_error",
    model_name="regression",
)


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL")
print("=" * 70)

label_encoder = LabelEncoder()

y_cls = label_encoder.fit_transform(event_target)

y_cls_train = y_cls[:split_index]
y_cls_test = y_cls[split_index:]

class_names = list(label_encoder.classes_)

classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

classifier.fit(X_train, y_cls_train)

cls_predictions = classifier.predict(X_test)

cls_accuracy = accuracy_score(y_cls_test, cls_predictions)
cls_f1 = f1_score(
    y_cls_test,
    cls_predictions,
    average="macro",
)

print(f"\nAccuracy: {cls_accuracy:.4f}")
print(f"Macro F1: {cls_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_cls_test,
        cls_predictions,
        target_names=class_names,
        zero_division=0,
    )
)

cls_importance = calculate_permutation_importance(
    classifier,
    X_test,
    y_cls_test,
    scoring="f1_macro",
    model_name="classification",
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HEAVY/EXTREME ERROR ANALYSIS")
print("=" * 70)

error_df = pd.DataFrame({
    "date": df.loc[
        valid_rows, "date"
    ].iloc[split_index:].values
    if "date" in df.columns
    else np.arange(len(y_reg_test)),

    "actual_rainfall": y_reg_test.values,

    "predicted_rainfall": reg_predictions,

    "actual_event": [
        class_names[value]
        for value in y_cls_test
    ],

    "predicted_event": [
        class_names[value]
        for value in cls_predictions
    ],
})

error_df["absolute_error"] = (
    error_df["actual_rainfall"]
    - error_df["predicted_rainfall"]
).abs()

error_df["signed_error"] = (
    error_df["predicted_rainfall"]
    - error_df["actual_rainfall"]
)

error_df["high_risk_actual"] = error_df["actual_event"].isin(
    ["Heavy Rain", "Extreme Rain"]
)

error_df["high_risk_missed"] = (
    error_df["high_risk_actual"]
    & ~error_df["predicted_event"].isin(
        ["Heavy Rain", "Extreme Rain"]
    )
)

high_risk_errors = error_df[
    error_df["high_risk_missed"]
].sort_values(
    "actual_rainfall",
    ascending=False,
)

high_risk_errors.to_csv(
    RESULT_DIR / "heavy_extreme_false_negatives.csv",
    index=False,
)

print(f"\nHigh-risk false negatives: {len(high_risk_errors)}")

if len(high_risk_errors) > 0:
    print("\nHigh-risk events predicted below Heavy Rain:")
    print(high_risk_errors.to_string(index=False))
else:
    print("No Heavy/Extreme false negatives found.")


# ============================================================
# EVENT-WISE REGRESSION MAE
# ============================================================

print("\nCalculating event-wise regression MAE...")

event_mae_rows = []

for event_name in class_names:
    mask = error_df["actual_event"] == event_name

    if mask.sum() == 0:
        mae = np.nan
    else:
        mae = mean_absolute_error(
            error_df.loc[mask, "actual_rainfall"],
            error_df.loc[mask, "predicted_rainfall"],
        )

    event_mae_rows.append({
        "event": event_name,
        "sample_count": int(mask.sum()),
        "mae": mae,
    })

event_mae_df = pd.DataFrame(event_mae_rows)

event_mae_df.to_csv(
    RESULT_DIR / "event_wise_mae.csv",
    index=False,
)

plt.figure(figsize=(9, 6))
plt.bar(
    event_mae_df["event"],
    event_mae_df["mae"],
)
plt.ylabel("Mean Absolute Error (mm)")
plt.xlabel("Actual rainfall event")
plt.title("Event-wise Rainfall Prediction Error")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "event_wise_mae.png",
    dpi=300,
)
plt.close()


# ============================================================
# ACTUAL VS PREDICTED RAINFALL
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    error_df["actual_rainfall"],
    error_df["predicted_rainfall"],
    alpha=0.6,
)

max_value = max(
    error_df["actual_rainfall"].max(),
    error_df["predicted_rainfall"].max(),
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
)

plt.xlabel("Actual next-day rainfall (mm)")
plt.ylabel("Predicted next-day rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "actual_vs_predicted_rainfall.png",
    dpi=300,
)
plt.close()


# ============================================================
# RAINFALL ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    error_df["signed_error"],
    bins=30,
)

plt.axvline(
    0,
    linestyle="--",
)

plt.xlabel("Prediction error (Predicted − Actual, mm)")
plt.ylabel("Frequency")
plt.title("Rainfall Prediction Error Distribution")

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "rainfall_error_distribution.png",
    dpi=300,
)
plt.close()


# ============================================================
# OPTIONAL SHAP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OPTIONAL SHAP ANALYSIS")
print("=" * 70)

try:
    import shap

    print("SHAP installed. Generating explainability plots...")

    # Use a subset to keep runtime manageable.
    shap_sample_size = min(150, len(X_test))

    X_shap = X_test.iloc[:shap_sample_size].copy()

    # ----------------------------
    # Regression SHAP
    # ----------------------------

    reg_explainer = shap.TreeExplainer(regressor)

    reg_shap_values = reg_explainer.shap_values(X_shap)

    plt.figure()
    shap.summary_plot(
        reg_shap_values,
        X_shap,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "shap_summary_regression.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------
    # Classification SHAP
    # ----------------------------

    cls_explainer = shap.TreeExplainer(classifier)

    cls_shap_values = cls_explainer.shap_values(X_shap)

    # For multiclass SHAP, generate one overall summary using
    # the mean absolute contribution across classes.
    if isinstance(cls_shap_values, list):
        cls_shap_array = np.stack(cls_shap_values, axis=-1)
        cls_shap_mean = np.mean(
            np.abs(cls_shap_array),
            axis=-1,
        )
    else:
        cls_shap_mean = cls_shap_values

    plt.figure()
    shap.summary_plot(
        cls_shap_mean,
        X_shap,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "shap_summary_classification.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("SHAP plots generated successfully.")

except ImportError:
    print(
        "\nSHAP is not installed."
        "\nSkipping SHAP analysis."
        "\nInstall it with: pip install shap"
    )

except Exception as e:
    print(f"\nSHAP analysis skipped because of an error: {e}")


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "model": "Regression",
        "mae": reg_mae,
        "rmse": reg_rmse,
        "r2": reg_r2,
        "accuracy": np.nan,
        "macro_f1": np.nan,
    },
    {
        "model": "Classification",
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "accuracy": cls_accuracy,
        "macro_f1": cls_f1,
    },
])

summary_df.to_csv(
    RESULT_DIR / "phase10_summary_metrics.csv",
    index=False,
)


# ============================================================
# TEXT SUMMARY
# ============================================================

top_regression_features = reg_importance.head(10)["feature"].tolist()
top_classification_features = cls_importance.head(10)["feature"].tolist()

with open(
    RESULT_DIR / "phase10_summary.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write("PHASE 10: MODEL EXPLAINABILITY SUMMARY\n")
    file.write("=" * 60 + "\n\n")

    file.write("Regression Performance\n")
    file.write(f"MAE: {reg_mae:.4f}\n")
    file.write(f"RMSE: {reg_rmse:.4f}\n")
    file.write(f"R2: {reg_r2:.4f}\n\n")

    file.write("Classification Performance\n")
    file.write(f"Accuracy: {cls_accuracy:.4f}\n")
    file.write(f"Macro F1: {cls_f1:.4f}\n\n")

    file.write("Top Regression Features\n")
    for feature in top_regression_features:
        file.write(f"- {feature}\n")

    file.write("\nTop Classification Features\n")
    for feature in top_classification_features:
        file.write(f"- {feature}\n")

    file.write(
        f"\nHeavy/Extreme false negatives: "
        f"{len(high_risk_errors)}\n"
    )

print("\n" + "=" * 70)
print("PHASE 10 COMPLETED")
print("=" * 70)

print(f"\nResults saved to:\n{RESULT_DIR}")

print("\nGenerated files:")
for path in sorted(RESULT_DIR.iterdir()):
    print(f"- {path.name}")