import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(r"F:\NavFusion\data\processed")
MODEL_DIR = Path(r"F:\NavFusion\models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("\nDataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# PYTORCH DATASETS
# ============================================================

# NumPy shape:
# (samples, time, features)
#
# Conv1D expects:
# (samples, channels, time)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).permute(0, 2, 1)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).permute(0, 2, 1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).permute(0, 2, 1)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
).unsqueeze(1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================

class SpeedCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=9,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.regressor(x)

        return x


model = SpeedCNN().to(device)

print("\nModel:")
print(model)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # -----------------------------
    # TRAIN
    # -----------------------------

    model.train()

    train_loss = 0.0

    for X, y in train_loader:

        X = X.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        prediction = model(X)

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * X.size(0)
        )

    train_loss /= len(train_dataset)

    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            prediction = model(X)

            loss = criterion(
                prediction,
                y
            )

            val_loss += (
                loss.item() * X.size(0)
            )

    val_loss /= len(val_dataset)

    val_rmse = val_loss ** 0.5
    train_rmse = train_loss ** 0.5

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train RMSE: {train_rmse:.4f} km/h | "
        f"Val RMSE: {val_rmse:.4f} km/h"
    )

    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_DIR / "speed_cnn_v1.pt"
        )

        print("  → Best model saved")

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    "Best model:",
    MODEL_DIR / "speed_cnn_v1.pt"
)