from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMU_FILE = PROJECT_ROOT / "results" / "imu_preprocessing.csv"
AI_FILE = PROJECT_ROOT / "results" / "ai_speed_output.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "imu_speed_correction.csv"


# ============================================================
# CONFIGURATION
# ============================================================

AI_MATCH_TOLERANCE_MS = 60

# Keep this deliberately small.
# We are NOT performing long-term accelerometer integration.
CORRECTION_GAIN = 1.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DAY 3 IMU + AI SPEED CORRECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading IMU data...")
    imu = pd.read_csv(IMU_FILE)

    print("Loading AI speed data...")
    ai = pd.read_csv(AI_FILE)

    print("\nIMU rows:", len(imu))
    print("AI rows :", len(ai))

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    required_imu = [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
        "linear_az",
        "linear_accel_magnitude",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "azimuth_deg",
        "pitch_deg",
        "roll_deg",
    ]

    required_ai = [
        "timestamp_ms",
        "ai_speed_mps",
        "speed_confidence",
    ]

    missing_imu = [
        c for c in required_imu
        if c not in imu.columns
    ]

    missing_ai = [
        c for c in required_ai
        if c not in ai.columns
    ]

    if missing_imu:
        raise ValueError(
            f"Missing IMU columns: {missing_imu}"
        )

    if missing_ai:
        raise ValueError(
            f"Missing AI columns: {missing_ai}"
        )

    # --------------------------------------------------------
    # SORT TIMESTAMPS
    # --------------------------------------------------------

    imu = (
        imu
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    ai = (
        ai
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CHECK TIMESTAMP QUALITY
    # --------------------------------------------------------

    if not imu["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("IMU timestamps are not ordered.")

    if not ai["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("AI timestamps are not ordered.")

    if imu["timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate IMU timestamps detected.")

    if ai["timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate AI timestamps detected.")

    # --------------------------------------------------------
    # ALIGN AI TO IMU
    # --------------------------------------------------------

    print("\nAligning AI predictions to IMU timestamps...")

    merged = pd.merge_asof(
        imu,
        ai,
        on="timestamp_ms",
        direction="nearest",
        tolerance=AI_MATCH_TOLERANCE_MS,
    )

    valid_ai = merged["ai_speed_mps"].notna()

    matched_count = int(valid_ai.sum())
    unmatched_count = int((~valid_ai).sum())

    print("AI/IMU matched   :", matched_count)
    print("AI/IMU unmatched :", unmatched_count)

    if matched_count == 0:
        raise ValueError(
            "No AI/IMU timestamps matched."
        )

    # --------------------------------------------------------
    # TIMESTAMP DT
    # --------------------------------------------------------

    merged["dt_s"] = (
        merged["timestamp_ms"]
        .diff()
        .div(1000.0)
    )

    merged["dt_s"] = (
        merged["dt_s"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # --------------------------------------------------------
    # IMU ACCELERATION
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We do NOT integrate this acceleration indefinitely.
    # The previous experiment demonstrated enormous drift.
    #
    # We use the acceleration only as a short-term correction
    # signal around the AI speed estimate.
    #
    # --------------------------------------------------------

    merged["imu_accel_mps2"] = (
        np.sqrt(
            merged["linear_ax"] ** 2
            + merged["linear_ay"] ** 2
        )
    )

    # --------------------------------------------------------
    # AI SPEED DERIVATIVE
    # --------------------------------------------------------

    ai_speed = merged["ai_speed_mps"]

    merged["ai_accel_mps2"] = np.nan

    matched_indices = merged.index[valid_ai]

    if len(matched_indices) > 1:

        matched_speed = (
            merged.loc[
                matched_indices,
                "ai_speed_mps"
            ]
        )

        matched_time = (
            merged.loc[
                matched_indices,
                "timestamp_ms"
            ]
        )

        speed_diff = matched_speed.diff()

        time_diff = (
            matched_time.diff() / 1000.0
        )

        ai_acceleration = (
            speed_diff / time_diff
        )

        ai_acceleration = (
            ai_acceleration
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0.0)
        )

        merged.loc[
            matched_indices,
            "ai_accel_mps2"
        ] = ai_acceleration

    # --------------------------------------------------------
    # RESIDUAL ACCELERATION
    # --------------------------------------------------------

    merged["accel_residual_mps2"] = np.nan

    merged.loc[
        valid_ai,
        "accel_residual_mps2"
    ] = (
        merged.loc[
            valid_ai,
            "imu_accel_mps2"
        ]
        -
        merged.loc[
            valid_ai,
            "ai_accel_mps2"
        ]
    )

    # --------------------------------------------------------
    # CONSERVATIVE SPEED CORRECTION
    # --------------------------------------------------------
    #
    # AI speed is the primary estimate.
    #
    # IMU provides only a small short-term correction.
    #
    # corrected_speed =
    #     AI speed +
    #     small correction from IMU residual
    #
    # --------------------------------------------------------

    merged["corrected_speed_mps"] = np.nan

    correction = (
        CORRECTION_GAIN
        * merged.loc[
            valid_ai,
            "accel_residual_mps2"
        ]
        * merged.loc[
            valid_ai,
            "dt_s"
        ]
    )

    merged.loc[
        valid_ai,
        "corrected_speed_mps"
    ] = (
        merged.loc[
            valid_ai,
            "ai_speed_mps"
        ]
        + correction
    )

    # Speed cannot be negative.
    merged.loc[
        valid_ai,
        "corrected_speed_mps"
    ] = (
        merged.loc[
            valid_ai,
            "corrected_speed_mps"
        ]
        .clip(lower=0.0)
    )

    # --------------------------------------------------------
    # VALIDATE MATCHED DATA ONLY
    # --------------------------------------------------------
    #
    # Unmatched IMU rows intentionally contain NaN in
    # AI-derived columns. Those are NOT errors.
    #
    # --------------------------------------------------------

    matched_numeric = (
        merged.loc[
            valid_ai
        ]
        .select_dtypes(include=[np.number])
    )

    if not np.isfinite(
        matched_numeric.to_numpy()
    ).all():

        raise ValueError(
            "Matched IMU + AI rows contain NaN or Inf."
        )

    print("\nMatched rows NaN/Inf: PASSED")

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    ai_mean = (
        merged.loc[
            valid_ai,
            "ai_speed_mps"
        ].mean()
    )

    ai_std = (
        merged.loc[
            valid_ai,
            "ai_speed_mps"
        ].std()
    )

    ai_min = (
        merged.loc[
            valid_ai,
            "ai_speed_mps"
        ].min()
    )

    ai_max = (
        merged.loc[
            valid_ai,
            "ai_speed_mps"
        ].max()
    )

    corrected_mean = (
        merged.loc[
            valid_ai,
            "corrected_speed_mps"
        ].mean()
    )

    corrected_std = (
        merged.loc[
            valid_ai,
            "corrected_speed_mps"
        ].std()
    )

    corrected_min = (
        merged.loc[
            valid_ai,
            "corrected_speed_mps"
        ].min()
    )

    corrected_max = (
        merged.loc[
            valid_ai,
            "corrected_speed_mps"
        ].max()
    )

    accel_mean = (
        merged.loc[
            valid_ai,
            "imu_accel_mps2"
        ].mean()
    )

    accel_std = (
        merged.loc[
            valid_ai,
            "imu_accel_mps2"
        ].std()
    )

    residual_mean = (
        merged.loc[
            valid_ai,
            "accel_residual_mps2"
        ].mean()
    )

    residual_std = (
        merged.loc[
            valid_ai,
            "accel_residual_mps2"
        ].std()
    )

    print(
        f"\nAI speed mean       : {ai_mean:.4f} m/s"
    )

    print(
        f"AI speed std        : {ai_std:.4f} m/s"
    )

    print(
        f"AI speed min        : {ai_min:.4f} m/s"
    )

    print(
        f"AI speed max        : {ai_max:.4f} m/s"
    )

    print(
        f"\nCorrected mean      : {corrected_mean:.4f} m/s"
    )

    print(
        f"Corrected std       : {corrected_std:.4f} m/s"
    )

    print(
        f"Corrected min       : {corrected_min:.4f} m/s"
    )

    print(
        f"Corrected max       : {corrected_max:.4f} m/s"
    )

    print(
        f"\nIMU accel mean      : {accel_mean:.4f} m/s²"
    )

    print(
        f"IMU accel std       : {accel_std:.4f} m/s²"
    )

    print(
        f"Residual accel mean : {residual_mean:.4f} m/s²"
    )

    print(
        f"Residual accel std  : {residual_std:.4f} m/s²"
    )

    print(
        f"\nCorrection gain     : {CORRECTION_GAIN}"
    )

    print(
        f"Matched samples     : {matched_count}"
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if (
        corrected_min < 0
        or not np.isfinite(corrected_mean)
        or not np.isfinite(corrected_max)
    ):
        raise ValueError(
            "Corrected speed validation failed."
        )

    print("\nCorrected speed validation: PASSED")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nOutput rows:", len(merged))

    print(
        "Saved to:"
    )

    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("DAY 3 IMU + AI SPEED CORRECTION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()