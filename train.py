import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("data/sample_weather.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Create target
df["heavy_rainfall"] = (
    df["rainfall_next_24h"] >= 64.5
).astype(int)

# Feature groups
rainfall_features = [
    "rainfall_mm"
]

atmospheric_features = [
    "temp_c",
    "humidity",
    "pressure",
    "wind_speed",
    "cape"
]

all_features = rainfall_features + atmospheric_features

# Remove missing values
df = df.dropna()

# Chronological split
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("\nDataset summary")
print("Total rows:", len(df))
print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Positive rate:", df["heavy_rainfall"].mean())

def evaluate_model(name, model, features):
    X_train = train_df[features]
    y_train = train_df["heavy_rainfall"]

    X_test = test_df[features]
    y_test = test_df["heavy_rainfall"]

    print("Training positives:", y_train.sum())
    print("Test positives:", y_test.sum())

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("-" * 30)

    if len(y_test.unique()) > 1:
        print("PR-AUC:", average_precision_score(y_test, probabilities))
        print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    # Try lower thresholds for rare-event detection
    for threshold in [0.5, 0.3, 0.2, 0.1]:
        predictions = (probabilities >= threshold).astype(int)

        print(f"\nThreshold: {threshold}")
        print("Predicted positives:", predictions.sum())
        print("Precision:", precision_score(y_test, predictions, zero_division=0))
        print("Recall:", recall_score(y_test, predictions, zero_division=0))
        print("F1:", f1_score(y_test, predictions, zero_division=0))

    # Final classification report using 0.3 threshold
    final_predictions = (probabilities >= 0.3).astype(int)

    print("\nClassification Report (Threshold = 0.3)")
    print(classification_report(
        y_test,
        final_predictions,
        zero_division=0
    ))
    X_train = train_df[features]
    y_train = train_df["heavy_rainfall"]

    X_test = test_df[features]
    y_test = test_df["heavy_rainfall"]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    print(f"\n{name}")
    print("-" * 30)
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

    if len(y_test.unique()) > 1:
        print("PR-AUC:", average_precision_score(y_test, probabilities))
        print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print(classification_report(y_test, predictions, zero_division=0))


# Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced"))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=3
    )
}


print("=== RAINFALL-ONLY EXPERIMENT ===")

for name, model in models.items():
    evaluate_model(
        name,
        model,
        rainfall_features
    )


print("\n=== RAINFALL + ATMOSPHERIC EXPERIMENT ===")

for name, model in models.items():
    evaluate_model(
        name,
        model,
        all_features
    )