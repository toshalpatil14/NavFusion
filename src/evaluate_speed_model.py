import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt


DATA_DIR = Path(r"D:\IDR-AI\data\processed")
MODEL_FILE = Path(r"D:\IDR-AI\models\speed_cnn_v1.pt")
RESULT_DIR = Path(r"D:\IDR-AI\results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


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

model = SpeedCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

print("\nTest data:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# ============================================================
# CONVERT
# ============================================================

X = torch.tensor(
    X_test,
    dtype=torch.float32
).permute(0, 2, 1)


# ============================================================
# PREDICT
# ============================================================

predictions = []

batch_size = 512

with torch.no_grad():

    for start in range(
        0,
        len(X),
        batch_size
    ):

        batch = X[
            start:start + batch_size
        ].to(device)

        output = model(batch)

        predictions.append(
            output.cpu().numpy().flatten()
        )


y_pred = np.concatenate(predictions)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(f"MAE  : {mae:.4f} km/h")
print(f"RMSE : {rmse:.4f} km/h")
print(f"R²   : {r2:.4f}")


# ============================================================
# ERROR
# ============================================================

errors = y_pred - y_test

print("\nError statistics:")
print(f"Mean error : {errors.mean():.4f} km/h")
print(f"Std error  : {errors.std():.4f} km/h")
print(f"Max error  : {np.abs(errors).max():.4f} km/h")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    RESULT_DIR / "speed_predictions.npy",
    y_pred
)

np.save(
    RESULT_DIR / "speed_ground_truth.npy",
    y_test
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    y_test,
    label="Ground Truth"
)

plt.plot(
    y_pred,
    label="AI Prediction"
)

plt.xlabel("Test Window")
plt.ylabel("Speed (km/h)")
plt.title("AI Speed Prediction — Test Set")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "speed_prediction_test.png",
    dpi=150
)

plt.close()


# ============================================================
# SCATTER
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    s=8,
    alpha=0.5
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Ground Truth Speed (km/h)")
plt.ylabel("Predicted Speed (km/h)")
plt.title("Ground Truth vs AI Prediction")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "speed_scatter_test.png",
    dpi=150
)

plt.close()


print("\nPlots saved to:")
print(RESULT_DIR)