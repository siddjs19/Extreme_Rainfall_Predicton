"""
PHASE 11: LEAKAGE-FREE MODEL IMPROVEMENT

Purpose:
Improve prediction of Heavy and Extreme Rainfall events
without using the rainfall value that directly defines the target.

Models:
1. Baseline Random Forest
2. Class-Weighted Random Forest
3. Oversampled Random Forest

Target:
Tomorrow's rainfall event.

Primary metrics:
- Accuracy
- Macro F1
- Heavy/Extreme recall
- Heavy/Extreme precision
- High-risk false negatives
"""
import joblib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "merged_imd_era5_pune_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "phase11_model_improvement"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# TARGET CONFIGURATION
# ============================================================

TARGET_REGRESSION = "rainfall_mm"

TARGET_TOMORROW = "target_rainfall_tomorrow"

TARGET_HEAVY = "target_heavy_rain"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def classify_rainfall(rainfall):
    """
    IMD-style rainfall categories consistent with Phase 10.
    """

    if rainfall <= 0:
        return "Dry"

    elif rainfall <= 15:
        return "Light Rain"

    elif rainfall <= 64.5:
        return "Moderate Rain"

    elif rainfall <= 115.5:
        return "Heavy Rain"

    else:
        return "Extreme Rain"


def calculate_high_risk_metrics(y_true, y_pred):
    """
    Heavy Rain and Extreme Rain are treated as high-risk events.
    """

    high_risk_classes = {
        "Heavy Rain",
        "Extreme Rain",
    }

    y_true_binary = np.array([
        int(label in high_risk_classes)
        for label in y_true
    ])

    y_pred_binary = np.array([
        int(label in high_risk_classes)
        for label in y_pred
    ])

    return {
        "high_risk_precision": precision_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        ),

        "high_risk_recall": recall_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        ),

        "high_risk_f1": f1_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        ),

        "high_risk_false_negatives": int(
            np.sum(
                (y_true_binary == 1)
                & (y_pred_binary == 0)
            )
        ),
    }


def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
    label_encoder,
):
    """
    Evaluate a classification model.
    """

    y_pred_encoded = model.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    high_risk_metrics = calculate_high_risk_metrics(
        y_test_labels,
        y_pred_labels,
    )

    metrics = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_test_labels,
            y_pred_labels,
        ),

        "macro_f1": f1_score(
            y_test_labels,
            y_pred_labels,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_test_labels,
            y_pred_labels,
            average="weighted",
            zero_division=0,
        ),

        **high_risk_metrics,
    }

    print(f"\n{model_name}")
    print("-" * len(model_name))

    print(
        classification_report(
            y_test_labels,
            y_pred_labels,
            zero_division=0,
        )
    )

    print("High-risk metrics:")
    print(
        f"Precision: {high_risk_metrics['high_risk_precision']:.4f}"
    )
    print(
        f"Recall:    {high_risk_metrics['high_risk_recall']:.4f}"
    )
    print(
        f"F1:        {high_risk_metrics['high_risk_f1']:.4f}"
    )
    print(
        f"False negatives: "
        f"{high_risk_metrics['high_risk_false_negatives']}"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test_labels,
        y_pred_labels,
        labels=label_encoder.classes_,
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    cm_df.to_csv(
        OUTPUT_DIR
        / f"confusion_matrix_{safe_name}.csv"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.colorbar()

    plt.xticks(
        range(len(label_encoder.classes_)),
        label_encoder.classes_,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(label_encoder.classes_)),
        label_encoder.classes_,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"confusion_matrix_{safe_name}.png",
        dpi=300,
    )

    plt.close()

    return metrics, y_test_labels, y_pred_labels


def save_high_risk_errors(
    model_name,
    y_true_labels,
    y_pred_labels,
    dates,
):
    """
    Save actual Heavy/Extreme events predicted as lower-risk classes.
    """

    high_risk_classes = {
        "Heavy Rain",
        "Extreme Rain",
    }

    rows = []

    for i in range(len(y_true_labels)):

        actual = y_true_labels[i]
        predicted = y_pred_labels[i]

        if (
            actual in high_risk_classes
            and predicted not in high_risk_classes
        ):
            rows.append({
                "date": dates.iloc[i],
                "actual_event": actual,
                "predicted_event": predicted,
            })

    errors_df = pd.DataFrame(rows)

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    errors_df.to_csv(
        OUTPUT_DIR
        / f"high_risk_false_negatives_{safe_name}.csv",
        index=False,
    )

    return errors_df


# ============================================================
# LOAD DATASET
# ============================================================

print_section("PHASE 11: LEAKAGE-FREE MODEL IMPROVEMENT")

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# PREPARE DATA
# ============================================================

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)


# ============================================================
# CREATE TOMORROW'S EVENT TARGET
# ============================================================

print_section("TARGET CREATION")

if TARGET_TOMORROW in df.columns:

    print(
        "Using existing target_rainfall_tomorrow column."
    )

    df["event_target"] = df[TARGET_TOMORROW].apply(
        classify_rainfall
    )

else:

    print(
        "target_rainfall_tomorrow not found."
    )

    print(
        "Creating tomorrow rainfall target using shifted rainfall_mm."
    )

    df["target_rainfall_tomorrow"] = (
        df[TARGET_REGRESSION].shift(-1)
    )

    df["event_target"] = (
        df[TARGET_TOMORROW].apply(classify_rainfall)
    )


# Remove final row because tomorrow's rainfall is unavailable
df = df.dropna(
    subset=["target_rainfall_tomorrow"]
).reset_index(drop=True)


# ============================================================
# DEFINE FEATURES WITHOUT TARGET LEAKAGE
# ============================================================

LEAKAGE_COLUMNS = [
    "time",

    # Today's rainfall is excluded because it directly reveals
    # current rainfall but is not the intended future prediction input.
    "rainfall_mm",

    # Future target
    "target_rainfall_tomorrow",

    # Existing binary target
    "target_heavy_rain",

    # Classification target
    "event_target",
]


feature_columns = [
    col
    for col in df.columns
    if col not in LEAKAGE_COLUMNS
]

X = df[feature_columns].copy()

# Keep numeric columns only
X = X.select_dtypes(include=[np.number])

# Replace invalid values
X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

# Median imputation
X = X.fillna(
    X.median(numeric_only=True)
)

y = df["event_target"].copy()


print(f"\nNumber of features: {X.shape[1]}")

print("Features:")
print(list(X.columns))


# ============================================================
# VERIFY NO LEAKAGE
# ============================================================

print_section("LEAKAGE CHECK")

for forbidden_column in LEAKAGE_COLUMNS:

    if forbidden_column in X.columns:
        raise RuntimeError(
            f"TARGET LEAKAGE DETECTED: {forbidden_column}"
        )

print("No target leakage detected.")

print(
    "\nClassification target distribution:"
)

print(y.value_counts())


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(df) * (1 - TEST_SIZE)
)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train_raw = y.iloc[:split_index].copy()
y_test_raw = y.iloc[split_index:].copy()

dates_test = df["time"].iloc[split_index:].copy()


print_section("TRAIN / TEST SPLIT")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining event distribution:")
print(y_train_raw.value_counts())

print("\nTesting event distribution:")
print(y_test_raw.value_counts())


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

print("\nClass labels:")
print(list(label_encoder.classes_))


# ============================================================
# MODEL TRAINING
# ============================================================

models = {}


# ------------------------------------------------------------
# MODEL 1: BASELINE
# ------------------------------------------------------------

print_section("MODEL 1: BASELINE RANDOM FOREST")

baseline_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None,
)

baseline_model.fit(
    X_train,
    y_train,
)

models["Baseline Random Forest"] = baseline_model


# ------------------------------------------------------------
# MODEL 2: CLASS WEIGHTED
# ------------------------------------------------------------

print_section("MODEL 2: CLASS-WEIGHTED RANDOM FOREST")

weighted_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

weighted_model.fit(
    X_train,
    y_train,
)
# Save trained class-weighted model for research demonstration app
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    weighted_model,
    MODEL_DIR / "rainfall_class_weighted_random_forest.pkl"
)

print(
    f"Saved trained model to: "
    f"{MODEL_DIR / 'rainfall_class_weighted_random_forest.pkl'}"
)
models["Class Weighted Random Forest"] = weighted_model


# ------------------------------------------------------------
# MODEL 3: OVERSAMPLING
# ------------------------------------------------------------

print_section("MODEL 3: RANDOM OVERSAMPLING")

oversampler = RandomOverSampler(
    random_state=RANDOM_STATE,
)

X_train_over, y_train_over = oversampler.fit_resample(
    X_train,
    y_train,
)

print(f"Original training size: {len(X_train)}")
print(f"Oversampled training size: {len(X_train_over)}")

print("\nOversampled class distribution:")
print(pd.Series(y_train_over).value_counts())


oversampled_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None,
)

oversampled_model.fit(
    X_train_over,
    y_train_over,
)

models["Oversampled Random Forest"] = oversampled_model


# ============================================================
# EVALUATION
# ============================================================

print_section("MODEL COMPARISON")

all_metrics = []
all_predictions = {}

for model_name, model in models.items():

    metrics, y_true_labels, y_pred_labels = evaluate_model(
        model_name,
        model,
        X_test,
        y_test,
        label_encoder,
    )

    all_metrics.append(metrics)

    all_predictions[model_name] = (
        y_true_labels,
        y_pred_labels,
    )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(all_metrics)

comparison_df = comparison_df.sort_values(
    by=[
        "high_risk_recall",
        "macro_f1",
    ],
    ascending=False,
)

comparison_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print_section("FINAL MODEL COMPARISON")

print(
    comparison_df.to_string(index=False)
)


# ============================================================
# HIGH-RISK ERROR ANALYSIS
# ============================================================

print_section("HIGH-RISK FALSE-NEGATIVE ANALYSIS")

error_summary = []

for model_name in models:

    y_true_labels, y_pred_labels = (
        all_predictions[model_name]
    )

    errors = save_high_risk_errors(
        model_name,
        y_true_labels,
        y_pred_labels,
        dates_test.reset_index(drop=True),
    )

    error_summary.append({
        "model": model_name,
        "high_risk_false_negatives": len(errors),
    })

    print(
        f"{model_name}: "
        f"{len(errors)} false negatives"
    )


error_summary_df = pd.DataFrame(error_summary)

error_summary_df.to_csv(
    OUTPUT_DIR / "high_risk_error_summary.csv",
    index=False,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print_section("FEATURE IMPORTANCE")

for model_name, model in models.items():

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    importance_df.to_csv(
        OUTPUT_DIR
        / f"feature_importance_{safe_name}.csv",
        index=False,
    )

    plt.figure(figsize=(10, 7))

    top_features = (
        importance_df.head(15)
        .sort_values("importance", ascending=True)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.title(
        f"Top Feature Importance: {model_name}"
    )

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"feature_importance_{safe_name}.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# BEST MODEL
# ============================================================

best_model_row = comparison_df.iloc[0]

best_model_name = best_model_row["model"]

print_section("BEST MODEL")

print(
    f"Selected based on Heavy/Extreme recall: "
    f"{best_model_name}"
)

print("\nMetrics:")
print(best_model_row.to_string())


# ============================================================
# SUMMARY FILE
# ============================================================

summary_lines = []

summary_lines.append(
    "PHASE 11: LEAKAGE-FREE MODEL IMPROVEMENT SUMMARY"
)

summary_lines.append("=" * 60)
summary_lines.append("")
summary_lines.append(
    f"Dataset: {DATA_PATH}"
)
summary_lines.append(
    f"Training samples: {len(X_train)}"
)
summary_lines.append(
    f"Testing samples: {len(X_test)}"
)
summary_lines.append(
    f"Number of features: {X.shape[1]}"
)
summary_lines.append("")
summary_lines.append(
    "Target: Tomorrow's rainfall event"
)
summary_lines.append(
    "Leakage columns excluded: rainfall_mm, "
    "target_rainfall_tomorrow, target_heavy_rain"
)
summary_lines.append("")
summary_lines.append("Models evaluated:")
summary_lines.append("- Baseline Random Forest")
summary_lines.append("- Class Weighted Random Forest")
summary_lines.append("- Oversampled Random Forest")
summary_lines.append("")
summary_lines.append(
    "Best model based on Heavy/Extreme recall:"
)
summary_lines.append(best_model_name)
summary_lines.append("")
summary_lines.append("MODEL COMPARISON")
summary_lines.append("-" * 60)
summary_lines.append(
    comparison_df.to_string(index=False)
)
summary_lines.append("")
summary_lines.append("Interpretation:")
summary_lines.append(
    "The primary objective is to reduce missed Heavy "
    "and Extreme Rainfall events."
)
summary_lines.append(
    "The corrected experiment excludes future information "
    "and rainfall values that directly define the target."
)
summary_lines.append(
    "Results should be compared against Phase 10 only "
    "after confirming identical target definitions."
)

with open(
    OUTPUT_DIR / "phase11_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(summary_lines))


# ============================================================
# COMPLETION
# ============================================================

print_section("PHASE 11 COMPLETED")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {file.name}")

print("\nNext step:")
print(
    "Review model_comparison.csv and determine whether "
    "class weighting or oversampling genuinely improves "
    "Heavy/Extreme Rainfall prediction."
)