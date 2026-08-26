import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path


# ============================================================
# AI SPEED -> NAVFUSION INTEGRATION OUTPUT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

NAVIGATION_FILE = PROJECT_ROOT / "results" / "navigation_interface_10hz.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ai_speed_output.csv"

WINDOW_SIZE = 50
STRIDE = 10
MC_DROPOUT_PASSES = 20

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
        return self.regressor(self.features(x))


def load_model(device):
    model = SpeedCNN().to(device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def enable_mc_dropout(model):
    """
    Keep BatchNorm layers in evaluation mode while enabling only Dropout.
    This gives a deterministic model prediction plus a Monte-Carlo dropout
    uncertainty estimate.

    IMPORTANT:
    speed_confidence is an uncertainty-derived proxy, not a calibrated
    probability of correctness. The current SpeedCNN has no dedicated
    confidence/uncertainty head.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def make_windows(df, scaler):
    X_raw = df[FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    windows = []
    timestamps = []

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE,
    ):
        end = start + WINDOW_SIZE

        windows.append(X_scaled[start:end])

        # Preserve the exact timestamp convention used by the
        # existing generate_speed_predictions.py:
        # timestamp of the LAST sample in each window.
        timestamps.append(
            int(df.iloc[end - 1][TIME_COLUMN])
        )

    X = np.asarray(windows, dtype=np.float32)

    return X, np.asarray(timestamps, dtype=np.int64)


def main():

    print("=" * 60)
    print("AI SPEED -> NAVFUSION INTEGRATION OUTPUT")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    for path in [
        DATA_FILE,
        SCALER_FILE,
        MODEL_FILE,
        NAVIGATION_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")
        print(f"OK: {path}")

    # --------------------------------------------------------
    # Load source dataset
    # --------------------------------------------------------

    print("\nLoading S-Vw4...")

    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
    )

    df.columns = df.columns.str.strip()

    missing = [
        col for col in FEATURES + [TIME_COLUMN]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df = (
        df.dropna(subset=FEATURES + [TIME_COLUMN])
        .reset_index(drop=True)
    )

    print(f"Rows after cleaning: {len(df)}")

    # --------------------------------------------------------
    # Load exact training scaler
    # --------------------------------------------------------

    scaler = joblib.load(SCALER_FILE)

    print("Exact training scaler loaded.")

    # --------------------------------------------------------
    # Create windows using the exact existing inference scheme
    # --------------------------------------------------------

    X, timestamps = make_windows(df, scaler)

    print(f"Windows: {X.shape}")
    print(f"Predictions: {len(timestamps)}")

    X_tensor = torch.from_numpy(
        X.transpose(0, 2, 1)
    ).to(device)

    # --------------------------------------------------------
    # Load exact trained model
    # --------------------------------------------------------

    model = load_model(device)
    model.eval()

    # --------------------------------------------------------
    # Deterministic prediction
    # --------------------------------------------------------

    deterministic_predictions_kmh = []

    with torch.no_grad():
        batch_size = 512

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):
            batch = X_tensor[start:start + batch_size]

            output = model(batch)

            deterministic_predictions_kmh.extend(
                output.squeeze(1)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    deterministic_predictions_kmh = np.asarray(
        deterministic_predictions_kmh,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Monte-Carlo dropout uncertainty
    # --------------------------------------------------------
    #
    # The model has Dropout(0.2), but no confidence head.
    # Therefore this is used only to derive a deterministic
    # uncertainty proxy. It is NOT a calibrated probability.
    # --------------------------------------------------------

    print(
        f"\nCalculating uncertainty proxy "
        f"({MC_DROPOUT_PASSES} MC-dropout passes)..."
    )

    mc_predictions = []

    enable_mc_dropout(model)

    with torch.no_grad():

        for pass_index in range(MC_DROPOUT_PASSES):

            pass_predictions = []

            batch_size = 512

            for start in range(
                0,
                len(X_tensor),
                batch_size,
            ):
                batch = X_tensor[start:start + batch_size]

                output = model(batch)

                pass_predictions.extend(
                    output.squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            mc_predictions.append(pass_predictions)

    mc_predictions = np.asarray(
        mc_predictions,
        dtype=np.float32,
    )

    predictive_std_kmh = mc_predictions.std(
        axis=0,
        ddof=1,
    )

    predictive_std_mps = predictive_std_kmh / 3.6

    # --------------------------------------------------------
    # Confidence proxy
    # --------------------------------------------------------
    #
    # Confidence is constrained to (0, 1].
    #
    # It is based on predictive uncertainty:
    #
    # confidence = 1 / (1 + sigma_mps)
    #
    # This is deliberately documented as an uncertainty-derived
    # proxy, NOT a calibrated probability.
    # --------------------------------------------------------

    speed_confidence = (
        1.0 /
        (1.0 + predictive_std_mps)
    )

    # --------------------------------------------------------
    # Convert model output km/h -> m/s
    # --------------------------------------------------------

    ai_speed_mps = (
        deterministic_predictions_kmh / 3.6
    )

    result = pd.DataFrame({
        "timestamp_ms": timestamps,
        "ai_speed_mps": ai_speed_mps,
        "speed_confidence": speed_confidence,
    })

    # --------------------------------------------------------
    # Validate exact requested schema
    # --------------------------------------------------------

    expected_columns = [
        "timestamp_ms",
        "ai_speed_mps",
        "speed_confidence",
    ]

    if result.columns.tolist() != expected_columns:
        raise RuntimeError(
            f"Unexpected output columns: {result.columns.tolist()}"
        )

    if result["timestamp_ms"].duplicated().any():
        raise RuntimeError(
            "Duplicate prediction timestamps detected."
        )

    # --------------------------------------------------------
    # Validate timestamp alignment against existing navigation
    # --------------------------------------------------------

    navigation = pd.read_csv(NAVIGATION_FILE)

    if "timestamp_ms" not in navigation.columns:
        raise ValueError(
            "navigation_interface_10hz.csv does not contain timestamp_ms."
        )

    navigation_timestamps = set(
        pd.to_numeric(
            navigation["timestamp_ms"],
            errors="coerce",
        )
        .dropna()
        .astype(np.int64)
        .tolist()
    )

    output_timestamps = set(
        result["timestamp_ms"].astype(np.int64)
    )

    exact_matches = len(
        output_timestamps.intersection(
            navigation_timestamps
        )
    )

    unmatched = len(
        output_timestamps.difference(
            navigation_timestamps
        )
    )

    match_rate = (
        100.0 * exact_matches / len(result)
        if len(result)
        else 0.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("AI SPEED OUTPUT CREATED")
    print("=" * 60)

    print(f"Predictions          : {len(result)}")
    print(
        f"First timestamp      : "
        f"{result['timestamp_ms'].iloc[0]}"
    )
    print(
        f"Last timestamp       : "
        f"{result['timestamp_ms'].iloc[-1]}"
    )

    print(
        f"Mean speed           : "
        f"{result['ai_speed_mps'].mean():.4f} m/s"
    )

    print(
        f"Min speed            : "
        f"{result['ai_speed_mps'].min():.4f} m/s"
    )

    print(
        f"Max speed            : "
        f"{result['ai_speed_mps'].max():.4f} m/s"
    )

    print(
        f"Mean confidence      : "
        f"{result['speed_confidence'].mean():.4f}"
    )

    print(
        f"Min confidence       : "
        f"{result['speed_confidence'].min():.4f}"
    )

    print(
        f"Max confidence       : "
        f"{result['speed_confidence'].max():.4f}"
    )

    print("\nTimestamp validation:")
    print(
        f"Exact matches        : "
        f"{exact_matches}"
    )
    print(
        f"Unmatched timestamps : "
        f"{unmatched}"
    )
    print(
        f"Match rate           : "
        f"{match_rate:.2f}%"
    )

    print("\nFirst 5 rows:")
    print(result.head().to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nNOTE:")
    print(
        "speed_confidence is an uncertainty-derived proxy "
        "from MC dropout, not a calibrated probability."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
