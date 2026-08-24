import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path(r"D:\IDR-AI\models\speed_cnn_v1.pt")
DATA_DIR = Path(r"D:\IDR-AI\data\processed")


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
# MODEL
# ============================================================

class SpeedCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(9, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
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

        return self.regressor(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = SpeedCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# TAKE ONE WINDOW
# ============================================================

sample = X_test[0]

print("\nOriginal sample shape:", sample.shape)


# Model expects:
# batch × channels × sequence
#
# Current:
# sequence × channels
#
# Therefore:
# 50 × 9 → 1 × 9 × 50

sample_tensor = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0).permute(0, 2, 1)


print("Model input shape:", sample_tensor.shape)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    prediction = model(
        sample_tensor.to(device)
    )

predicted_speed = prediction.item()

ground_truth = float(
    y_test[0]
)


# ============================================================
# RESULTS
# ============================================================

print("\n==============================")
print("SPEED PREDICTION")
print("==============================")

print(
    f"Ground truth : {ground_truth:.2f} km/h"
)

print(
    f"Prediction   : {predicted_speed:.2f} km/h"
)

print(
    f"Error        : {abs(predicted_speed - ground_truth):.2f} km/h"
)

print("==============================")