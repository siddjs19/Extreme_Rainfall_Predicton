from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged_imd_era5_pune_features.csv"
RESULTS = ROOT / "results" / "phase11_model_improvement"
OUT = ROOT / "results" / "report_visuals"

OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Rainfall-category distribution
# -----------------------------
df = pd.read_csv(DATA)

target = "target_rainfall_tomorrow"

def rainfall_category(value):
    if value <= 0:
        return "Dry"
    elif value <= 15:
        return "Light"
    elif value <= 64.5:
        return "Moderate"
    elif value <= 115.5:
        return "Heavy"
    return "Extreme"

categories = df[target].apply(rainfall_category)
counts = categories.value_counts().reindex(
    ["Dry", "Light", "Moderate", "Heavy", "Extreme"],
    fill_value=0
)

plt.figure(figsize=(8, 5))
counts.plot(kind="bar")
plt.title("Distribution of Next-Day Rainfall Categories")
plt.xlabel("Rainfall Category")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "rainfall_category_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 2. Model comparison
# -----------------------------
comparison_file = RESULTS / "model_comparison.csv"

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)

    metric_columns = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "high_risk_precision",
        "high_risk_recall",
        "high_risk_f1"
    ]

    available_metrics = [
        col for col in metric_columns if col in comparison.columns
    ]

    if available_metrics:
        ax = comparison.set_index("model")[available_metrics].plot(
            kind="bar",
            figsize=(11, 6)
        )

        ax.set_title("Comparison of Rainfall Classification Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(OUT / "model_comparison.png", dpi=300)
        plt.close()

# -----------------------------
# 3. High-risk recall comparison
# -----------------------------
if comparison_file.exists() and "high_risk_recall" in comparison.columns:
    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["model"],
        comparison["high_risk_recall"]
    )

    plt.title("High-Risk Rainfall Detection Recall")
    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "high_risk_recall_comparison.png", dpi=300)
    plt.close()

# -----------------------------
# 4. Copy/reference existing Phase 11 visuals
# -----------------------------
existing_visuals = [
    "confusion_matrix_baseline_random_forest.png",
    "confusion_matrix_class_weighted_random_forest.png",
    "confusion_matrix_oversampled_random_forest.png",
    "feature_importance_baseline_random_forest.png",
    "feature_importance_class_weighted_random_forest.png",
    "feature_importance_oversampled_random_forest.png",
]

for filename in existing_visuals:
    source = RESULTS / filename
    destination = OUT / filename

    if source.exists():
        destination.write_bytes(source.read_bytes())

print(f"Visuals created in: {OUT}")
for file in sorted(OUT.iterdir()):
    print(file.name)