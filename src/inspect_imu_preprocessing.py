from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = OUTPUT_DIR / "imu_preprocessing.csv"


# ============================================================
# COLUMN NAMES
# ============================================================

TIMESTAMP_COL = "TIME SINCE START (ms)"

ACC_X_COL = "ACCELEROMETER X (m/s²)"
ACC_Y_COL = "ACCELEROMETER Y (m/s²)"
ACC_Z_COL = "ACCELEROMETER Z (m/s²)"

GRAV_X_COL = "GRAVITY X (m/s²)"
GRAV_Y_COL = "GRAVITY Y (m/s²)"
GRAV_Z_COL = "GRAVITY Z (m/s²)"

GYRO_X_COL = "GYROSCOPE X (rad/s)"
GYRO_Y_COL = "GYROSCOPE Y (rad/s)"
GYRO_Z_COL = "GYROSCOPE Z (rad/s)"

AZIMUTH_COL = "ORIENTATION (Azimuth) (Â°)"
PITCH_COL = "ORIENTATION (Pitch) (Â°)"
ROLL_COL = "ORIENTATION (Roll ) (Â°)"


# ============================================================
# HELPER
# ============================================================

def print_stats(name, values):
    values = np.asarray(values, dtype=float)

    print(f"\n{name}")
    print(f"  Min  : {np.min(values):.6f}")
    print(f"  Mean : {np.mean(values):.6f}")
    print(f"  Max  : {np.max(values):.6f}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMU PREPROCESSING DIAGNOSTIC")
    print("=" * 60)

    print("\nLoading sensor data...")
    print("Input:", INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        encoding="latin1"
    )

    df.columns = df.columns.str.strip()

    required_columns = [
        TIMESTAMP_COL,

        ACC_X_COL,
        ACC_Y_COL,
        ACC_Z_COL,

        GRAV_X_COL,
        GRAV_Y_COL,
        GRAV_Z_COL,

        GYRO_X_COL,
        GYRO_Y_COL,
        GYRO_Z_COL,

        AZIMUTH_COL,
        PITCH_COL,
        ROLL_COL,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    print("Rows:", len(df))
    print("Required columns: PASSED")


    # ========================================================
    # EXTRACT NUMERIC DATA
    # ========================================================

    timestamp_ms = pd.to_numeric(
        df[TIMESTAMP_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    acc_x = pd.to_numeric(
        df[ACC_X_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    acc_y = pd.to_numeric(
        df[ACC_Y_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    acc_z = pd.to_numeric(
        df[ACC_Z_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    grav_x = pd.to_numeric(
        df[GRAV_X_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    grav_y = pd.to_numeric(
        df[GRAV_Y_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    grav_z = pd.to_numeric(
        df[GRAV_Z_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    gyro_x = pd.to_numeric(
        df[GYRO_X_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    gyro_y = pd.to_numeric(
        df[GYRO_Y_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    gyro_z = pd.to_numeric(
        df[GYRO_Z_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    azimuth = pd.to_numeric(
        df[AZIMUTH_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    pitch = pd.to_numeric(
        df[PITCH_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    roll = pd.to_numeric(
        df[ROLL_COL],
        errors="coerce"
    ).to_numpy(dtype=float)


    # ========================================================
    # VALIDATE FINITE INPUT
    # ========================================================

    input_arrays = np.column_stack([
        timestamp_ms,

        acc_x,
        acc_y,
        acc_z,

        grav_x,
        grav_y,
        grav_z,

        gyro_x,
        gyro_y,
        gyro_z,

        azimuth,
        pitch,
        roll,
    ])

    if np.isnan(input_arrays).any():
        raise ValueError("Validation failed: NaN values found.")

    if not np.isfinite(input_arrays).all():
        raise ValueError("Validation failed: Inf values found.")

    print("NaN check: PASSED")
    print("Inf check: PASSED")


    # ========================================================
    # TIMESTAMP VALIDATION
    # ========================================================

    dt_ms = np.diff(timestamp_ms)

    if np.any(dt_ms <= 0):
        raise ValueError(
            "Validation failed: timestamps are not strictly ordered."
        )

    dt_s = dt_ms / 1000.0

    print("\n" + "=" * 60)
    print("TIMESTAMP VALIDATION")
    print("=" * 60)

    print(f"First timestamp : {timestamp_ms[0]:.0f} ms")
    print(f"Last timestamp  : {timestamp_ms[-1]:.0f} ms")
    print(f"Rows            : {len(timestamp_ms)}")

    print(f"\ndt Min    : {np.min(dt_s):.6f} s")
    print(f"dt Mean   : {np.mean(dt_s):.6f} s")
    print(f"dt Median : {np.median(dt_s):.6f} s")
    print(f"dt Max    : {np.max(dt_s):.6f} s")

    print("\nTimestamp ordering: PASSED")


    # ========================================================
    # GRAVITY COMPENSATION
    # ========================================================

    linear_ax = acc_x - grav_x
    linear_ay = acc_y - grav_y
    linear_az = acc_z - grav_z

    linear_accel_magnitude = np.sqrt(
        linear_ax ** 2
        + linear_ay ** 2
        + linear_az ** 2
    )


    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    output_arrays = np.column_stack([
        linear_ax,
        linear_ay,
        linear_az,
        linear_accel_magnitude,
    ])

    if np.isnan(output_arrays).any():
        raise ValueError(
            "Validation failed: NaN found after gravity compensation."
        )

    if not np.isfinite(output_arrays).all():
        raise ValueError(
            "Validation failed: Inf found after gravity compensation."
        )


    # ========================================================
    # PRINT STATISTICS
    # ========================================================

    print("\n" + "=" * 60)
    print("GRAVITY-COMPENSATED LINEAR ACCELERATION")
    print("=" * 60)

    print_stats("Linear Acceleration X (m/s²)", linear_ax)
    print_stats("Linear Acceleration Y (m/s²)", linear_ay)
    print_stats("Linear Acceleration Z (m/s²)", linear_az)
    print_stats(
        "Linear Acceleration Magnitude (m/s²)",
        linear_accel_magnitude
    )


    print("\n" + "=" * 60)
    print("GYROSCOPE")
    print("=" * 60)

    print_stats("Gyroscope X (rad/s)", gyro_x)
    print_stats("Gyroscope Y (rad/s)", gyro_y)
    print_stats("Gyroscope Z (rad/s)", gyro_z)


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = pd.DataFrame({
        "timestamp_ms": timestamp_ms.astype(np.int64),

        "linear_ax": linear_ax,
        "linear_ay": linear_ay,
        "linear_az": linear_az,

        "linear_accel_magnitude": (
            linear_accel_magnitude
        ),

        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,

        "azimuth_deg": azimuth,
        "pitch_deg": pitch,
        "roll_deg": roll,
    })

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("IMU PREPROCESSING COMPLETE")
    print("=" * 60)

    print("Output rows:", len(output))
    print("No NaN/Inf: PASSED")
    print("Timestamp order: PASSED")
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()