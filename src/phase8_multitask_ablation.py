import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42
EPOCHS = 250
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HEAVY_THRESHOLD = 64.5

DATA_PATH = "data/merged_imd_era5_pune_features.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Event classification
# ============================================================

def rainfall_event(rainfall):
    if rainfall == 0:
        return 0          # Dry
    elif rainfall <= 15:
        return 1          # Light
    elif rainfall <= 64.5:
        return 2          # Moderate
    elif rainfall <= 115.5:
        return 3          # Heavy
    else:
        return 4          # Extreme


# ============================================================
# Multi-task neural network
# ============================================================

class MultiTaskWeatherNet(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.heavy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.event_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        shared_features = self.shared(x)

        rainfall = self.regression_head(shared_features)
        heavy_logits = self.heavy_head(shared_features)
        event_logits = self.event_head(shared_features)

        return rainfall, heavy_logits, event_logits


# ============================================================
# Load and prepare data
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Tomorrow's rainfall target
    df["target_rainfall_tomorrow"] = df["rainfall_mm"].shift(-1)

    # Tomorrow's heavy-rain target
    df["target_heavy_rain_tomorrow"] = (
        df["target_rainfall_tomorrow"] > HEAVY_THRESHOLD
    ).astype(int)

    # Tomorrow's five-category event target
    df["target_event_tomorrow"] = (
        df["target_rainfall_tomorrow"].apply(rainfall_event)
    )

    # Remove final row because tomorrow's rainfall is unavailable
    df = df.dropna(
        subset=[
            "target_rainfall_tomorrow",
            "target_heavy_rain_tomorrow",
            "target_event_tomorrow",
        ]
    ).reset_index(drop=True)

    # Remove target columns and date
    drop_columns = [
        "date",
        "target_rainfall_tomorrow",
        "target_heavy_rain_tomorrow",
        "target_event_tomorrow",
    ]

    # Exclude direct ERA5 precipitation to avoid duplication/leakage
    drop_columns += [
        col for col in df.columns
        if "total_precip" in col.lower()
    ]

    feature_columns = [
        col for col in df.columns
        if col not in drop_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y_reg = df["target_rainfall_tomorrow"].values.astype(np.float32)

    y_heavy = df["target_heavy_rain_tomorrow"].values.astype(np.float32)

    y_event = df["target_event_tomorrow"].values.astype(np.int64)

    return df, X, y_reg, y_heavy, y_event, feature_columns


# ============================================================
# Feature-group selection
# ============================================================

def classify_feature(feature):

    name = feature.lower()

    # Calendar and IMD-derived features
    if any(term in name for term in [
        "rainfall",
        "rain_",
        "lag",
        "rolling",
        "roll",
        "month",
        "day",
        "season",
        "year",
        "monsoon",
        "winter",
        "summer",
        "post_monsoon",
        "pre_monsoon",
        "dry",
    ]):
        return "imd"

    # Temperature
    if any(term in name for term in [
        "temperature",
        "temp",
        "skin_temp",
        "soil_temp",
    ]):
        return "temperature"

    # Moisture / humidity
    if any(term in name for term in [
        "dewpoint",
        "dew_point",
        "humidity",
        "relative_humidity",
        "rh",
        "vapor",
        "moisture",
    ]):
        return "moisture"

    # Pressure
    if any(term in name for term in [
        "pressure",
        "surface_pressure",
        "msl",
    ]):
        return "pressure"

    # Wind
    if any(term in name for term in [
        "wind",
        "u10",
        "v10",
        "gust",
    ]):
        return "wind"

    # Remaining ERA5 variables
    return "other"


def get_feature_groups(feature_columns):

    groups = {
        "IMD Only": [],
        "IMD + Temperature": [],
        "IMD + Moisture": [],
        "IMD + Pressure": [],
        "IMD + Wind": [],
        "IMD + All ERA5": [],
    }

    for feature in feature_columns:

        category = classify_feature(feature)

        # All IMD features are included in every experiment
        if category == "imd":
            groups["IMD Only"].append(feature)
            groups["IMD + Temperature"].append(feature)
            groups["IMD + Moisture"].append(feature)
            groups["IMD + Pressure"].append(feature)
            groups["IMD + Wind"].append(feature)
            groups["IMD + All ERA5"].append(feature)

        elif category == "temperature":
            groups["IMD + Temperature"].append(feature)
            groups["IMD + All ERA5"].append(feature)

        elif category == "moisture":
            groups["IMD + Moisture"].append(feature)
            groups["IMD + All ERA5"].append(feature)

        elif category == "pressure":
            groups["IMD + Pressure"].append(feature)
            groups["IMD + All ERA5"].append(feature)

        elif category == "wind":
            groups["IMD + Wind"].append(feature)
            groups["IMD + All ERA5"].append(feature)

        else:
            # Other ERA5 variables are included only in the complete model
            groups["IMD + All ERA5"].append(feature)

    return groups


# ============================================================
# Train one multi-task model
# ============================================================

def train_model(
    X_train_scaled,
    y_reg_train,
    y_heavy_train,
    y_event_train,
    input_size,
):
    model = MultiTaskWeatherNet(input_size)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_reg_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_heavy_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_event_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    regression_loss = nn.SmoothL1Loss()

    heavy_classes = np.array([0, 1])

    heavy_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=heavy_classes,
        y=y_heavy_train.astype(int),
    )

    pos_weight = torch.tensor(
        heavy_class_weights[1] / heavy_class_weights[0],
        dtype=torch.float32,
    )

    heavy_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    event_classes = np.unique(y_event_train)

    event_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=event_classes,
        y=y_event_train,
    )

    event_weight_tensor = torch.ones(
        5,
        dtype=torch.float32,
    )

    for cls, weight in zip(
        event_classes,
        event_class_weights,
    ):
        event_weight_tensor[int(cls)] = float(weight)

    event_loss = nn.CrossEntropyLoss(
        weight=event_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        total_reg_loss = 0
        total_heavy_loss = 0
        total_event_loss = 0

        for (
            batch_x,
            batch_y_reg,
            batch_y_heavy,
            batch_y_event,
        ) in train_loader:

            optimizer.zero_grad()

            pred_reg, pred_heavy, pred_event = model(batch_x)

            loss_reg = regression_loss(
                pred_reg,
                batch_y_reg,
            )

            loss_heavy = heavy_loss(
                pred_heavy,
                batch_y_heavy,
            )

            loss_event = event_loss(
                pred_event,
                batch_y_event,
            )

            loss = (
                loss_reg
                + loss_heavy
                + loss_event
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            total_reg_loss += loss_reg.item()
            total_heavy_loss += loss_heavy.item()
            total_event_loss += loss_event.item()

        avg_loss = total_loss / len(train_loader)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "regression_loss": (
                total_reg_loss / len(train_loader)
            ),
            "heavy_loss": (
                total_heavy_loss / len(train_loader)
            ),
            "event_loss": (
                total_event_loss / len(train_loader)
            ),
        })

        if avg_loss < best_loss:
            best_loss = avg_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Total: {avg_loss:.4f} | "
                f"Reg: {total_reg_loss / len(train_loader):.4f} | "
                f"Heavy: {total_heavy_loss / len(train_loader):.4f} | "
                f"Event: {total_event_loss / len(train_loader):.4f}"
            )

    model.load_state_dict(best_state)

    return model, history


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test_scaled,
    y_reg_test,
    y_heavy_test,
    y_event_test,
):

    model.eval()

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred_reg_tensor, pred_heavy_tensor, pred_event_tensor = model(
            X_test_tensor
        )

    pred_rainfall = (
        pred_reg_tensor.numpy().flatten()
    )

    pred_heavy_probability = (
        torch.sigmoid(pred_heavy_tensor)
        .numpy()
        .flatten()
    )

    pred_heavy_class = (
        pred_heavy_probability >= 0.5
    ).astype(int)

    pred_event_class = (
        torch.argmax(pred_event_tensor, dim=1)
        .numpy()
    )

    metrics = {
        "rainfall_mae": float(
            mean_absolute_error(
                y_reg_test,
                pred_rainfall,
            )
        ),
        "rainfall_rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_reg_test,
                    pred_rainfall,
                )
            )
        ),
        "rainfall_r2": float(
            r2_score(
                y_reg_test,
                pred_rainfall,
            )
        ),
        "heavy_accuracy": float(
            accuracy_score(
                y_heavy_test,
                pred_heavy_class,
            )
        ),
        "heavy_f1": float(
            f1_score(
                y_heavy_test,
                pred_heavy_class,
                zero_division=0,
            )
        ),
        "heavy_roc_auc": float(
            roc_auc_score(
                y_heavy_test,
                pred_heavy_probability,
            )
        ),
        "event_accuracy": float(
            accuracy_score(
                y_event_test,
                pred_event_class,
            )
        ),
        "event_macro_f1": float(
            f1_score(
                y_event_test,
                pred_event_class,
                average="macro",
                zero_division=0,
            )
        ),
        "event_weighted_f1": float(
            f1_score(
                y_event_test,
                pred_event_class,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    return metrics


# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 8: MULTI-TASK ABLATION STUDY")
    print("=" * 70)

    df, X, y_reg, y_heavy, y_event, feature_columns = load_data()

    split_idx = int(len(df) * 0.80)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_reg_train = y_reg[:split_idx]
    y_reg_test = y_reg[split_idx:]

    y_heavy_train = y_heavy[:split_idx]
    y_heavy_test = y_heavy[split_idx:]

    y_event_train = y_event[:split_idx]
    y_event_test = y_event[split_idx:]

    feature_groups = get_feature_groups(feature_columns)

    # Print selected feature counts
    print("\nFeature groups:")

    for experiment, features in feature_groups.items():
        print(
            f"{experiment}: "
            f"{len(features)} features"
        )

    results = []

    for experiment, selected_features in feature_groups.items():

        print("\n" + "=" * 70)
        print(f"RUNNING: {experiment}")
        print("=" * 70)

        set_seed(SEED)

        # Select features
        X_train_group = X_train[selected_features]
        X_test_group = X_test[selected_features]

        # Standardize using training data only
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_group
        )

        X_test_scaled = scaler.transform(
            X_test_group
        )

        # Train
        model, history = train_model(
            X_train_scaled,
            y_reg_train,
            y_heavy_train,
            y_event_train,
            input_size=len(selected_features),
        )

        # Evaluate
        metrics = evaluate_model(
            model,
            X_test_scaled,
            y_reg_test,
            y_heavy_test,
            y_event_test,
        )

        result = {
            "experiment": experiment,
            "feature_count": len(selected_features),
            **metrics,
        }

        results.append(result)

        print("\nResults:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Save individual model
        safe_name = (
            experiment.lower()
            .replace(" ", "_")
            .replace("+", "plus")
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                RESULTS_DIR,
                f"phase8_{safe_name}_model.pt",
            ),
        )

        # Save individual training history
        pd.DataFrame(history).to_csv(
            os.path.join(
                RESULTS_DIR,
                f"phase8_{safe_name}_history.csv",
            ),
            index=False,
        )

    # ========================================================
    # Save final comparison
    # ========================================================

    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        RESULTS_DIR,
        "phase8_multitask_ablation_results.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # Save feature groups
    feature_group_json = {
        experiment: features
        for experiment, features in feature_groups.items()
    }

    with open(
        os.path.join(
            RESULTS_DIR,
            "phase8_feature_groups.json",
        ),
        "w",
    ) as f:
        json.dump(
            feature_group_json,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FINAL ABLATION COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "experiment",
                "feature_count",
                "rainfall_mae",
                "rainfall_rmse",
                "rainfall_r2",
                "heavy_f1",
                "event_accuracy",
                "event_macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f" - {results_path}")
    print(" - results/phase8_feature_groups.json")
    print(" - Individual model and training-history files")


if __name__ == "__main__":
    main()