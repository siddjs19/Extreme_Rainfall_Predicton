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
    classification_report,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility
# -----------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------
# Paths
# -----------------------------

DATA_PATH = "data/merged_imd_era5_pune_features.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -----------------------------
# Create tomorrow targets
# -----------------------------

df["target_rainfall_tomorrow"] = df["rainfall_mm"].shift(-1)

# Heavy rainfall threshold used in earlier phases
HEAVY_THRESHOLD = 64.5

df["target_heavy_rain_tomorrow"] = (
    df["target_rainfall_tomorrow"] > HEAVY_THRESHOLD
).astype(int)


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


df["target_event_tomorrow"] = (
    df["target_rainfall_tomorrow"].apply(rainfall_event)
)


# Remove final row because tomorrow's target is unavailable
df = df.dropna(
    subset=[
        "target_rainfall_tomorrow",
        "target_heavy_rain_tomorrow",
        "target_event_tomorrow",
    ]
).reset_index(drop=True)


# -----------------------------
# Select predictors
# -----------------------------

TARGET_COLUMNS = [
    "target_rainfall_tomorrow",
    "target_heavy_rain_tomorrow",
    "target_event_tomorrow",
]

DROP_COLUMNS = [
    "time",
    "rainfall_mm",
    "event_target",
]

# Exclude direct ERA5 precipitation to avoid duplication/leakage
DROP_COLUMNS += [
    col for col in df.columns
    if "total_precip" in col.lower()
]


FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in DROP_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]


X = df[FEATURE_COLUMNS].copy()
y_reg = df["target_rainfall_tomorrow"].values.astype(np.float32)
y_heavy = df["target_heavy_rain_tomorrow"].values.astype(np.float32)
y_event = df["target_event_tomorrow"].values.astype(np.int64)


# Fill missing values using training-compatible median strategy
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())


# -----------------------------
# Chronological train/test split
# -----------------------------

split_idx = int(len(df) * 0.80)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_reg_train = y_reg[:split_idx]
y_reg_test = y_reg[split_idx:]

y_heavy_train = y_heavy[:split_idx]
y_heavy_test = y_heavy[split_idx:]

y_event_train = y_event[:split_idx]
y_event_test = y_event[split_idx:]


# -----------------------------
# Standardize features
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Convert to PyTorch tensors
# -----------------------------

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_reg_train_tensor = torch.tensor(
    y_reg_train,
    dtype=torch.float32
).view(-1, 1)

y_heavy_train_tensor = torch.tensor(
    y_heavy_train,
    dtype=torch.float32
).view(-1, 1)

y_event_train_tensor = torch.tensor(
    y_event_train,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_reg_train_tensor,
    y_heavy_train_tensor,
    y_event_train_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)


# -----------------------------
# Multi-task neural network
# -----------------------------

class MultiTaskWeatherNet(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # Task 1: rainfall regression
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Task 2: heavy rainfall classification
        self.heavy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Task 3: five-class rainfall event classification
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


model = MultiTaskWeatherNet(
    input_size=X_train_scaled.shape[1]
)


# -----------------------------
# Loss functions
# -----------------------------

regression_loss = nn.SmoothL1Loss()

# Heavy rainfall imbalance weighting
classes = np.array([0, 1])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_heavy_train.astype(int),
)

pos_weight = torch.tensor(
    class_weights[1] / class_weights[0],
    dtype=torch.float32,
)

heavy_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# Event imbalance weighting
event_classes = np.unique(y_event_train)

event_weights = compute_class_weight(
    class_weight="balanced",
    classes=event_classes,
    y=y_event_train,
)

event_weight_tensor = torch.ones(5, dtype=torch.float32)

for cls, weight in zip(event_classes, event_weights):
    event_weight_tensor[int(cls)] = float(weight)

event_loss = nn.CrossEntropyLoss(
    weight=event_weight_tensor
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)


# -----------------------------
# Training
# -----------------------------

EPOCHS = 250

history = []

best_loss = float("inf")
best_state = None

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
            batch_y_reg
        )

        loss_heavy = heavy_loss(
            pred_heavy,
            batch_y_heavy
        )

        loss_event = event_loss(
            pred_event,
            batch_y_event
        )

        # Equal multi-task weighting
        loss = (
            loss_reg
            + loss_heavy
            + loss_event
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
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
        "regression_loss": total_reg_loss / len(train_loader),
        "heavy_loss": total_heavy_loss / len(train_loader),
        "event_loss": total_event_loss / len(train_loader),
    })

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = {
            key: value.cpu().clone()
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


# Restore best training state
model.load_state_dict(best_state)


# -----------------------------
# Evaluation
# -----------------------------

model.eval()

with torch.no_grad():
    pred_reg_tensor, pred_heavy_tensor, pred_event_tensor = model(
        X_test_tensor
    )

pred_rainfall = pred_reg_tensor.numpy().flatten()

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


# -----------------------------
# Metrics
# -----------------------------

regression_metrics = {
    "MAE": float(mean_absolute_error(y_reg_test, pred_rainfall)),
    "RMSE": float(
        np.sqrt(
            mean_squared_error(y_reg_test, pred_rainfall)
        )
    ),
    "R2": float(r2_score(y_reg_test, pred_rainfall)),
}

heavy_metrics = {
    "accuracy": float(
        accuracy_score(y_heavy_test, pred_heavy_class)
    ),
    "f1": float(
        f1_score(y_heavy_test, pred_heavy_class, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_heavy_test, pred_heavy_probability)
    ),
}

event_metrics = {
    "accuracy": float(
        accuracy_score(y_event_test, pred_event_class)
    ),
    "macro_f1": float(
        f1_score(
            y_event_test,
            pred_event_class,
            average="macro",
            zero_division=0,
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_event_test,
            pred_event_class,
            average="weighted",
            zero_division=0,
        )
    ),
}


print("\n" + "=" * 60)
print("PHASE 7: MULTI-TASK LEARNING RESULTS")
print("=" * 60)

print("\nTask 1 — Tomorrow Rainfall Regression")
for key, value in regression_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nTask 2 — Tomorrow Heavy Rain Classification")
for key, value in heavy_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nTask 3 — Tomorrow Multi-Event Classification")
for key, value in event_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMulti-Event Classification Report:")
print(
    classification_report(
        y_event_test,
        pred_event_class,
        target_names=[
            "Dry",
            "Light",
            "Moderate",
            "Heavy",
            "Extreme",
        ],
        zero_division=0,
    )
)


# -----------------------------
# Save results
# -----------------------------

results = {
    "phase": "Phase 7 - Multi-Task Learning",
    "model": "Shared neural representation with three task-specific heads",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "feature_count": int(len(FEATURE_COLUMNS)),
    "features": FEATURE_COLUMNS,
    "target_definitions": {
        "regression": "Rainfall amount on the following day",
        "heavy_classification": f"Tomorrow rainfall > {HEAVY_THRESHOLD} mm",
        "event_classification": (
            "0=Dry, 1=Light, 2=Moderate, "
            "3=Heavy, 4=Extreme"
        ),
    },
    "regression": regression_metrics,
    "heavy_rain_classification": heavy_metrics,
    "multi_event_classification": event_metrics,
    "training_history": history,
}


with open(
    os.path.join(
        RESULTS_DIR,
        "phase7_multitask_results.json"
    ),
    "w",
) as f:
    json.dump(results, f, indent=2)


predictions = pd.DataFrame({
    "date": df.iloc[split_idx:]["date"].values,
    "actual_rainfall_tomorrow": y_reg_test,
    "predicted_rainfall_tomorrow": pred_rainfall,
    "actual_heavy_rain": y_heavy_test,
    "predicted_heavy_probability": pred_heavy_probability,
    "predicted_heavy_class": pred_heavy_class,
    "actual_event": y_event_test,
    "predicted_event": pred_event_class,
})

predictions.to_csv(
    os.path.join(
        RESULTS_DIR,
        "phase7_multitask_predictions.csv"
    ),
    index=False,
)


pd.DataFrame(history).to_csv(
    os.path.join(
        RESULTS_DIR,
        "phase7_training_history.csv"
    ),
    index=False,
)


torch.save(
    model.state_dict(),
    os.path.join(
        RESULTS_DIR,
        "phase7_multitask_model.pt"
    ),
)

print("\nSaved:")
print(" - results/phase7_multitask_results.json")
print(" - results/phase7_multitask_predictions.csv")
print(" - results/phase7_training_history.csv")
print(" - results/phase7_multitask_model.pt")