import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    r"F:\IO-VNBD-DATA\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset\S-Dataset\S-Vw2.csv"
)

MODEL_FILE = Path(
    r"F:\NavFusion\models\speed_cnn_v1.pt"
)

SCALER_FILE = Path(
    r"F:\NavFusion\data\processed\imu_scaler.pkl"
)


RESULT_DIR = Path(
    r"F:\NavFusion\results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GYROSCOPE X (rad/s)",
    "GYROSCOPE Y (rad/s)",
    "GYROSCOPE Z (rad/s)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)"
]

TARGET = "GPS SPEED (Kmh)"

WINDOW_SIZE = 50


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==============================")
print("UNSEEN DRIVE EVALUATION")
print("==============================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

if not SCALER_FILE.exists():
    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_FILE}"
    )

print("Dataset: OK")
print("Model:   OK")
print("Scaler:  OK")


# ============================================================
# MODEL
# ============================================================

class SpeedCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                9,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                1
            )
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
# LOAD SCALER
# ============================================================

print("\nLoading training scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded successfully.")

print(
    "Scaler features:",
    scaler.n_features_in_
)

if scaler.n_features_in_ != 9:

    raise ValueError(
        f"Expected scaler with 9 features, "
        f"but found {scaler.n_features_in_}"
    )


# ============================================================
# LOAD UNSEEN DATA
# ============================================================

print("\nLoading unseen drive:")

print(DATA_FILE)

df = pd.read_csv(
    DATA_FILE,
    encoding="latin1"
)

df.columns = df.columns.str.strip()

print("\nRows loaded:", len(df))

print("Columns:", len(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nMissing columns:")

    for column in missing_columns:
        print(column)

    raise ValueError(
        "Required dataset columns are missing."
    )


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning data...")

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print(
    "Rows after cleaning:",
    len(df)
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_raw = df[
    FEATURES
].astype(float)

y_raw = df[
    TARGET
].astype(float).values


print("\nRaw feature shape:")
print(X_raw.shape)

print("\nGround truth speed:")
print(
    f"Min  : {y_raw.min():.2f} km/h"
)

print(
    f"Max  : {y_raw.max():.2f} km/h"
)

print(
    f"Mean : {y_raw.mean():.2f} km/h"
)


# ============================================================
# APPLY TRAINING SCALER
# ============================================================

print("\nApplying training scaler...")

# IMPORTANT:
# We ONLY transform.
#
# We do NOT fit the scaler on S-Vw2.
#
# This keeps S-Vw2 completely unseen.

X_scaled = scaler.transform(
    X_raw
)

print(
    "Scaled shape:",
    X_scaled.shape
)


# ============================================================
# CREATE WINDOWS
# ============================================================

print("\nCreating windows...")

X_windows = []
y_windows = []

for i in range(
    WINDOW_SIZE - 1,
    len(X_scaled)
):

    window = X_scaled[
        i - WINDOW_SIZE + 1:
        i + 1
    ]

    target = y_raw[i]

    X_windows.append(
        window
    )

    y_windows.append(
        target
    )


X_windows = np.asarray(
    X_windows,
    dtype=np.float32
)

y_windows = np.asarray(
    y_windows,
    dtype=np.float32
)


print(
    "Window dataset:"
)

print(
    "X:",
    X_windows.shape
)

print(
    "y:",
    y_windows.shape
)


# ============================================================
# CONVERT TO TORCH
# ============================================================

# Current:
#
# X_windows =
# samples × 50 × 9
#
# CNN expects:
#
# samples × 9 × 50

X_tensor = torch.tensor(
    X_windows,
    dtype=torch.float32
).permute(
    0,
    2,
    1
)


print(
    "CNN input:",
    X_tensor.shape
)


# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning CNN inference...")

predictions = []

batch_size = 512

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        batch_size
    ):

        end = start + batch_size

        batch = X_tensor[
            start:end
        ].to(device)

        output = model(
            batch
        )

        predictions.append(
            output
            .cpu()
            .numpy()
            .flatten()
        )


y_pred = np.concatenate(
    predictions
)


# ============================================================
# BASIC PREDICTION CHECK
# ============================================================

print("\nPrediction statistics:")

print(
    f"Min prediction  : {y_pred.min():.2f} km/h"
)

print(
    f"Max prediction  : {y_pred.max():.2f} km/h"
)

print(
    f"Mean prediction : {y_pred.mean():.2f} km/h"
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_windows,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_windows,
        y_pred
    )
)

r2 = r2_score(
    y_windows,
    y_pred
)


# ============================================================
# ERROR
# ============================================================

errors = (
    y_pred -
    y_windows
)

mean_error = errors.mean()

std_error = errors.std()

max_error = np.abs(
    errors
).max()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("==============================")
print("UNSEEN DRIVE RESULTS")
print("==============================")

print(
    "Drive: S-Vw2.csv"
)

print(
    f"Samples: {len(y_windows)}"
)

print()

print(
    f"MAE  : {mae:.4f} km/h"
)

print(
    f"RMSE : {rmse:.4f} km/h"
)

print(
    f"R²   : {r2:.4f}"
)

print()

print(
    "ERROR STATISTICS"
)

print(
    f"Mean error : {mean_error:.4f} km/h"
)

print(
    f"Std error  : {std_error:.4f} km/h"
)

print(
    f"Max error  : {max_error:.4f} km/h"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = (
    RESULT_DIR /
    "S-Vw2_metrics.txt"
)

with open(
    metrics_file,
    "w"
) as f:

    f.write(
        "Speed CNN V1 - Unseen Drive\n"
    )

    f.write(
        "==============================\n"
    )

    f.write(
        "Drive: S-Vw2.csv\n"
    )

    f.write(
        f"Samples: {len(y_windows)}\n\n"
    )

    f.write(
        f"MAE: {mae:.6f} km/h\n"
    )

    f.write(
        f"RMSE: {rmse:.6f} km/h\n"
    )

    f.write(
        f"R2: {r2:.6f}\n\n"
    )

    f.write(
        f"Mean error: {mean_error:.6f} km/h\n"
    )

    f.write(
        f"Std error: {std_error:.6f} km/h\n"
    )

    f.write(
        f"Max error: {max_error:.6f} km/h\n"
    )


# ============================================================
# SAVE PREDICTIONS CSV
# ============================================================

result_df = pd.DataFrame({

    "ground_truth_kmh":
        y_windows,

    "predicted_kmh":
        y_pred,

    "error_kmh":
        errors

})

prediction_file = (
    RESULT_DIR /
    "S-Vw2_predictions.csv"
)

result_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# GROUND TRUTH VS PREDICTION
# ============================================================

print("\nCreating prediction plot...")

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    y_windows,
    label="Ground Truth",
    linewidth=1
)

plt.plot(
    y_pred,
    label="AI Prediction",
    linewidth=1
)

plt.xlabel(
    "Window"
)

plt.ylabel(
    "Speed (km/h)"
)

plt.title(
    "Speed CNN V1 - Unseen Drive S-Vw2"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

prediction_plot = (
    RESULT_DIR /
    "S-Vw2_prediction.png"
)

plt.savefig(
    prediction_plot,
    dpi=150
)

plt.close()


# ============================================================
# SCATTER PLOT
# ============================================================

print(
    "Creating scatter plot..."
)

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_windows,
    y_pred,
    s=8,
    alpha=0.5
)

minimum = min(
    y_windows.min(),
    y_pred.min()
)

maximum = max(
    y_windows.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)

plt.xlabel(
    "Ground Truth Speed (km/h)"
)

plt.ylabel(
    "Predicted Speed (km/h)"
)

plt.title(
    "S-Vw2 Ground Truth vs AI Prediction"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

scatter_plot = (
    RESULT_DIR /
    "S-Vw2_scatter.png"
)

plt.savefig(
    scatter_plot,
    dpi=150
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("==============================")
print("UNSEEN DRIVE TEST COMPLETE")
print("==============================")

print("\nFiles saved:")

print(
    prediction_file
)

print(
    metrics_file
)

print(
    prediction_plot
)

print(
    scatter_plot
)