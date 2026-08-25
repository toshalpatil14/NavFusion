import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\IDR-AI")

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)

SCALER_FILE = PROJECT_ROOT / "data" / "processed" / "imu_scaler.pkl"

MODEL_FILE = PROJECT_ROOT / "models" / "speed_cnn_v1.pt"

OUTPUT_FILE = PROJECT_ROOT / "results" / "S-Vw4_speed_predictions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 50
STRIDE = 10

FEATURES = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE X (rad/s)",
    "GYROSCOPE Y (rad/s)",
    "GYROSCOPE Z (rad/s)",
    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",
]

TIME_COLUMN = "TIME SINCE START (ms)"
TARGET_COLUMN = "GPS SPEED (Kmh)"


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

            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.regressor(x)

        return x


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SPEED CNN → SPEED INTERFACE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking files...")

    for path in [DATA_FILE, SCALER_FILE, MODEL_FILE]:

        if not path.exists():
            raise FileNotFoundError(f"File not found:\n{path}")

        print(f"OK: {path}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_FILE, encoding="latin1")

    df.columns = df.columns.str.strip()

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = FEATURES + [
        TIME_COLUMN,
        TARGET_COLUMN,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURES + [TIME_COLUMN, TARGET_COLUMN]
    ).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)}")

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    print("\nLoading scaler...")

    scaler = joblib.load(SCALER_FILE)

    print("Scaler loaded.")

    # --------------------------------------------------------
    # Scale IMU features
    # --------------------------------------------------------

    X_raw = df[FEATURES].values.astype(np.float32)

    X_scaled = scaler.transform(X_raw)

    print(f"Scaled feature shape: {X_scaled.shape}")

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    print("\nCreating windows...")

    windows = []
    timestamps = []
    targets = []

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        window = X_scaled[start:end]

        # Timestamp of LAST sample in window
        timestamp = int(
            df.iloc[end - 1][TIME_COLUMN]
        )

        # Target of LAST sample in window
        target = float(
            df.iloc[end - 1][TARGET_COLUMN]
        )

        windows.append(window)
        timestamps.append(timestamp)
        targets.append(target)

    X = np.asarray(windows, dtype=np.float32)

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64
    )

    targets = np.asarray(
        targets,
        dtype=np.float32
    )

    print(f"Windows: {X.shape}")
    print(f"Timestamps: {timestamps.shape}")

    # --------------------------------------------------------
    # Convert to PyTorch format
    # --------------------------------------------------------

    # Current:
    # (windows, 50, 9)
    #
    # CNN requires:
    # (windows, 9, 50)

    X_tensor = torch.from_numpy(
        X.transpose(0, 2, 1)
    ).to(device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Speed CNN...")

    model = SpeedCNN().to(device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = []

    with torch.no_grad():

        batch_size = 512

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ]

            output = model(batch)

            predictions.extend(
                output.squeeze(1)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Create interface dataframe
    # --------------------------------------------------------

    result = pd.DataFrame({

        "timestamp_ms": timestamps,

        "estimated_speed_kmh": predictions,

    })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SPEED INTERFACE CREATED")
    print("=" * 60)

    print(f"Predictions: {len(result)}")

    print(
        f"First timestamp: "
        f"{result['timestamp_ms'].iloc[0]}"
    )

    print(
        f"Last timestamp: "
        f"{result['timestamp_ms'].iloc[-1]}"
    )

    print(
        f"Mean predicted speed: "
        f"{result['estimated_speed_kmh'].mean():.3f} km/h"
    )

    print(
        f"Min predicted speed: "
        f"{result['estimated_speed_kmh'].min():.3f} km/h"
    )

    print(
        f"Max predicted speed: "
        f"{result['estimated_speed_kmh'].max():.3f} km/h"
    )

    print("\nFirst 5 predictions:")

    print(result.head())

    print("\nSaved to:")

    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()