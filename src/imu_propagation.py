"""
Day 3 - First-stage IMU propagation.

Uses:
    - gravity-compensated linear acceleration
    - gyroscope Z
    - measured azimuth

This is a standalone experiment.
It does NOT modify the production EKF.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "imu_preprocessing.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "imu_propagation.csv"


def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def main():

    print("=" * 60)
    print("DAY 3 IMU PROPAGATION")
    print("=" * 60)

    print("\nLoading:", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
        "linear_az",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "azimuth_deg",
        "pitch_deg",
        "roll_deg",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not df["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("Timestamps are not ordered.")

    # ------------------------------------------------------------
    # TIMESTEP
    # ------------------------------------------------------------

    df["dt_s"] = df["timestamp_ms"].diff().fillna(0.0) / 1000.0

    # ------------------------------------------------------------
    # IMU ACCELERATION
    # ------------------------------------------------------------

    # Horizontal acceleration magnitude.
    #
    # We intentionally do not assume that the phone X/Y axes
    # directly equal vehicle forward/lateral axes.
    df["horizontal_accel_mps2"] = np.sqrt(
        df["linear_ax"] ** 2 +
        df["linear_ay"] ** 2
    )

    # ------------------------------------------------------------
    # GYRO YAW INTEGRATION
    # ------------------------------------------------------------

    yaw_gyro = np.zeros(len(df), dtype=float)

    if len(df) > 0:
        yaw_gyro[0] = df["azimuth_deg"].iloc[0]

    for i in range(1, len(df)):

        dt = float(df["dt_s"].iloc[i])

        # Smartphone gyro Z is used as the yaw-rate signal.
        yaw_gyro[i] = yaw_gyro[i - 1] + math.degrees(
            float(df["gyro_z"].iloc[i]) * dt
        )

    # Keep gyro-only yaw numerically bounded.
    yaw_gyro = np.array(
        [value % 360.0 for value in yaw_gyro],
        dtype=float
    )

    df["gyro_yaw_deg"] = yaw_gyro

    # ------------------------------------------------------------
    # AZIMUTH + 180 DEG VEHICLE HEADING
    # ------------------------------------------------------------

    df["vehicle_heading_deg"] = (
        df["azimuth_deg"] + 180.0
    ) % 360.0

    # ------------------------------------------------------------
    # FIRST-STAGE ACCELERATION INTEGRATION
    # ------------------------------------------------------------

    velocity = 0.0
    distance = 0.0

    velocities = np.zeros(len(df), dtype=float)
    distances = np.zeros(len(df), dtype=float)

    for i in range(1, len(df)):

        dt = float(df["dt_s"].iloc[i])

        acceleration = float(
            df["horizontal_accel_mps2"].iloc[i]
        )

        # Simple first-stage magnitude integration.
        velocity += acceleration * dt
        distance += velocity * dt

        velocities[i] = velocity
        distances[i] = distance

    df["imu_velocity_mps"] = velocities
    df["imu_distance_m"] = distances

    # ------------------------------------------------------------
    # SANITY CHECKS
    # ------------------------------------------------------------

    numeric = df.select_dtypes(include=[np.number])

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("IMU propagation produced NaN or Inf.")

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("IMU PROPAGATION COMPLETE")
    print("=" * 60)

    print(f"Rows                 : {len(df)}")
    print(
        f"Mean dt              : {df['dt_s'].mean():.6f} s"
    )
    print(
        f"Horizontal accel mean: "
        f"{df['horizontal_accel_mps2'].mean():.6f} m/s²"
    )
    print(
        f"IMU velocity final   : "
        f"{df['imu_velocity_mps'].iloc[-1]:.3f} m/s"
    )
    print(
        f"IMU distance final   : "
        f"{df['imu_distance_m'].iloc[-1]:.3f} m"
    )

    print("\nNo NaN/Inf: PASSED")
    print("Timestamp order: PASSED")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()