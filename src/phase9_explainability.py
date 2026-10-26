from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\research\WeatherGPT-Research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "phase9_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "merged_imd_era5_pune_features.csv"

PHASE6_PREDICTIONS = (
    RESULTS_DIR / "phase6_multi_event_forecasting_predictions.csv"
)

PHASE7_PREDICTIONS = (
    RESULTS_DIR / "phase7_multitask_predictions.csv"
)

EVENT_LABELS = [
    "Dry",
    "Light",
    "Moderate",
    "Heavy",
    "Extreme"
]


# ============================================================
# HELPERS
# ============================================================

def save_figure(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def normalize_event_labels(series):
    """
    Converts numeric event labels to names if necessary.
    Leaves existing string labels unchanged.
    """

    numeric_to_event = {
        0: "Dry",
        1: "Light",
        2: "Moderate",
        3: "Heavy",
        4: "Extreme"
    }

    # Handle numeric labels
    if pd.api.types.is_numeric_dtype(series):
        return series.map(numeric_to_event)

    # Handle string labels safely
    return (
        series.astype(str)
        .str.strip()
        .str.title()
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])


# ============================================================
# PHASE 6: NEXT-DAY EVENT CONFUSION MATRIX
# ============================================================

if PHASE6_PREDICTIONS.exists():

    p6 = pd.read_csv(PHASE6_PREDICTIONS)

    print("\nPhase 6 columns:")
    print(p6.columns.tolist())

    # Actual column names from your Phase 6 CSV
    y_true = normalize_event_labels(
        p6["actual_event_tomorrow"]
    )

    y_pred = normalize_event_labels(
        p6["predicted_event_tomorrow"]
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=EVENT_LABELS
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EVENT_LABELS
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    ax.set_title(
        "Phase 6: Next-Day Rainfall Event Confusion Matrix",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Event")
    ax.set_ylabel("Actual Event")

    save_figure("phase6_confusion_matrix.png")

    print("\nPhase 6 Classification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=EVENT_LABELS,
            target_names=EVENT_LABELS,
            zero_division=0
        )
    )

else:
    print(
        f"\nPhase 6 prediction file not found:\n"
        f"{PHASE6_PREDICTIONS}"
    )


# ============================================================
# PHASE 7: MULTITASK EVENT CONFUSION MATRIX
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    print("\nPhase 7 columns:")
    print(p7.columns.tolist())

    # Actual column names from your Phase 7 CSV
    y_true = normalize_event_labels(
        p7["actual_event"]
    )

    y_pred = normalize_event_labels(
        p7["predicted_event"]
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=EVENT_LABELS
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EVENT_LABELS
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    ax.set_title(
        "Phase 7: Multitask Next-Day Rainfall Event Confusion Matrix",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Event")
    ax.set_ylabel("Actual Event")

    save_figure("phase7_confusion_matrix.png")

    print("\nPhase 7 Classification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=EVENT_LABELS,
            target_names=EVENT_LABELS,
            zero_division=0
        )
    )

else:
    print(
        f"\nPhase 7 prediction file not found:\n"
        f"{PHASE7_PREDICTIONS}"
    )


# ============================================================
# FALSE-NEGATIVE ANALYSIS
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    # Normalize event labels first
    actual_events = normalize_event_labels(
        p7["actual_event"]
    )

    predicted_events = normalize_event_labels(
        p7["predicted_event"]
    )

    # Heavy and Extreme incorrectly predicted as
    # Dry, Light, or Moderate.
    high_risk_actual = actual_events.isin(
        ["Heavy", "Extreme"]
    )

    missed_high_risk = (
        high_risk_actual
        & ~predicted_events.isin(["Heavy", "Extreme"])
    )

    missed = p7[missed_high_risk].copy()

    missed["actual_event"] = actual_events[missed_high_risk]
    missed["predicted_event"] = predicted_events[missed_high_risk]

    print(
        "\nHigh-risk false negatives:",
        len(missed)
    )

    if len(missed) > 0:

        print("\nMissed Heavy/Extreme Events:")

        display_columns = [
            col for col in [
                "date",
                "actual_rainfall_tomorrow",
                "predicted_rainfall_tomorrow",
                "actual_event",
                "predicted_event"
            ]
            if col in missed.columns
        ]

        print(
            missed[display_columns].head(20).to_string(index=False)
        )

        missed.to_csv(
            OUTPUT_DIR / "heavy_extreme_false_negatives.csv",
            index=False
        )

        print(
            f"\nSaved: "
            f"{OUTPUT_DIR / 'heavy_extreme_false_negatives.csv'}"
        )

        # ----------------------------------------------------
        # Plot missed Heavy/Extreme rainfall cases
        # ----------------------------------------------------

        if (
            "actual_rainfall_tomorrow" in missed.columns
            and "predicted_rainfall_tomorrow" in missed.columns
        ):

            plt.figure(figsize=(10, 5))

            x = np.arange(len(missed))

            plt.bar(
                x - 0.2,
                missed["actual_rainfall_tomorrow"],
                width=0.4,
                label="Actual Rainfall"
            )

            plt.bar(
                x + 0.2,
                missed["predicted_rainfall_tomorrow"],
                width=0.4,
                label="Predicted Rainfall"
            )

            plt.xlabel("Missed High-Risk Cases")
            plt.ylabel("Rainfall (mm)")

            plt.title(
                "Heavy/Extreme Rainfall False Negatives",
                fontsize=14,
                fontweight="bold"
            )

            plt.legend()

            save_figure(
                "heavy_extreme_false_negatives.png"
            )

    else:
        print(
            "No Heavy/Extreme false negatives found."
        )


# ============================================================
# ACTUAL VS PREDICTED RAINFALL
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    actual_column = "actual_rainfall_tomorrow"
    predicted_column = "predicted_rainfall_tomorrow"

    if (
        actual_column in p7.columns
        and predicted_column in p7.columns
    ):

        actual = p7[actual_column].values
        predicted = p7[predicted_column].values

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        r2 = r2_score(
            actual,
            predicted
        )

        plt.figure(figsize=(8, 7))

        plt.scatter(
            actual,
            predicted,
            alpha=0.65
        )

        max_value = max(
            actual.max(),
            predicted.max()
        )

        plt.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
            linewidth=2,
            label="Perfect Prediction"
        )

        plt.xlabel("Actual Rainfall (mm)")
        plt.ylabel("Predicted Rainfall (mm)")

        plt.title(
            "Multitask Model: Actual vs Predicted Next-Day Rainfall",
            fontsize=14,
            fontweight="bold"
        )

        plt.text(
            0.05,
            0.95,
            f"MAE = {mae:.2f} mm\n"
            f"RMSE = {rmse:.2f} mm\n"
            f"R² = {r2:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

        plt.legend()

        save_figure(
            "actual_vs_predicted_rainfall.png"
        )


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    actual_column = "actual_rainfall_tomorrow"
    predicted_column = "predicted_rainfall_tomorrow"

    if (
        actual_column in p7.columns
        and predicted_column in p7.columns
    ):

        errors = (
            p7[predicted_column]
            - p7[actual_column]
        )

        plt.figure(figsize=(8, 5))

        plt.hist(
            errors,
            bins=30
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=2
        )

        plt.xlabel("Prediction Error (mm)")
        plt.ylabel("Frequency")

        plt.title(
            "Distribution of Rainfall Prediction Errors",
            fontsize=14,
            fontweight="bold"
        )

        save_figure(
            "rainfall_error_distribution.png"
        )


# ============================================================
# EVENT-WISE MAE
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    actual_column = "actual_rainfall_tomorrow"
    predicted_column = "predicted_rainfall_tomorrow"

    if (
        actual_column in p7.columns
        and predicted_column in p7.columns
        and "actual_event" in p7.columns
    ):

        p7["_actual_event_normalized"] = normalize_event_labels(
            p7["actual_event"]
        )

        event_mae = []

        for event_name in EVENT_LABELS:

            subset = p7[
                p7["_actual_event_normalized"] == event_name
            ]

            if len(subset) > 0:

                value = mean_absolute_error(
                    subset[actual_column],
                    subset[predicted_column]
                )

                event_mae.append(value)

            else:
                event_mae.append(np.nan)

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            EVENT_LABELS,
            event_mae
        )

        plt.xlabel("Actual Rainfall Event")
        plt.ylabel("Mean Absolute Error (mm)")

        plt.title(
            "Rainfall Prediction Error by Event Category",
            fontsize=14,
            fontweight="bold"
        )

        for bar, value in zip(bars, event_mae):

            if not np.isnan(value):

                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom"
                )

        save_figure(
            "event_wise_mae.png"
        )


# ============================================================
# SAVE SUMMARY METRICS
# ============================================================

if PHASE7_PREDICTIONS.exists():

    p7 = pd.read_csv(PHASE7_PREDICTIONS)

    actual_column = "actual_rainfall_tomorrow"
    predicted_column = "predicted_rainfall_tomorrow"

    if (
        actual_column in p7.columns
        and predicted_column in p7.columns
    ):

        actual_events = normalize_event_labels(
            p7["actual_event"]
        )

        predicted_events = normalize_event_labels(
            p7["predicted_event"]
        )

        summary = pd.DataFrame({
            "Metric": [
                "MAE",
                "RMSE",
                "R2",
                "Event Accuracy",
                "Event Macro F1"
            ],
            "Value": [
                mean_absolute_error(
                    p7[actual_column],
                    p7[predicted_column]
                ),
                np.sqrt(
                    mean_squared_error(
                        p7[actual_column],
                        p7[predicted_column]
                    )
                ),
                r2_score(
                    p7[actual_column],
                    p7[predicted_column]
                ),
                np.mean(
                    actual_events == predicted_events
                ),
                classification_report(
                    actual_events,
                    predicted_events,
                    labels=EVENT_LABELS,
                    target_names=EVENT_LABELS,
                    output_dict=True,
                    zero_division=0
                )["macro avg"]["f1-score"]
            ]
        })

        summary.to_csv(
            OUTPUT_DIR / "phase9_summary_metrics.csv",
            index=False
        )

        print(
            f"Saved: "
            f"{OUTPUT_DIR / 'phase9_summary_metrics.csv'}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\nPhase 9 completed.")
print(f"All outputs saved in:\n{OUTPUT_DIR}")