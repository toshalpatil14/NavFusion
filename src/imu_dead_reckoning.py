import math
from pathlib import Path

import numpy as np
import pandas as pd

from coordinates import xy_to_latlon
from ins import propagate_2d


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

IMU_FILE = RESULTS_DIR / "imu_frame_transform.csv"
SPEED_FILE = RESULTS_DIR / "imu_speed_correction.csv"
AI_FILE = RESULTS_DIR / "ai_speed_output.csv"
OUTPUT_FILE = RESULTS_DIR / "imu_dead_reckoning.csv"


def finite_check(df, columns, name):
    values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def main():
    print("=" * 60)
    print("IMU DEAD RECKONING")
    print("=" * 60)

    print("Loading IMU frame-transform data...")
    imu = pd.read_csv(IMU_FILE)

    print("Loading speed-correction data...")
    speed = pd.read_csv(SPEED_FILE)

    print("Loading AI speed data...")
    ai = pd.read_csv(AI_FILE)

    print(f"IMU rows   : {len(imu)}")
    print(f"Speed rows : {len(speed)}")
    print(f"AI rows    : {len(ai)}")

    # ---------------------------------------------------------
    # Validate IMU input
    # ---------------------------------------------------------

    required_imu = [
        "timestamp_ms",
        "travel_heading_deg",
        "accel_forward_mps2",
        "accel_lateral_mps2",
        "accel_east_mps2",
        "accel_north_mps2",
    ]

    missing = [c for c in required_imu if c not in imu.columns]

    if missing:
        raise ValueError(
            "Missing IMU columns: " + ", ".join(missing)
        )

    if imu["timestamp_ms"].duplicated().any():
        raise ValueError("IMU timestamps contain duplicates.")

    if not imu["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("IMU timestamps are not ordered.")

    finite_check(
        imu,
        [
            "timestamp_ms",
            "travel_heading_deg",
            "accel_forward_mps2",
            "accel_lateral_mps2",
            "accel_east_mps2",
            "accel_north_mps2",
        ],
        "IMU input",
    )

    # ---------------------------------------------------------
    # Prepare speed sources
    # ---------------------------------------------------------

    speed["timestamp_ms"] = pd.to_numeric(
        speed["timestamp_ms"], errors="coerce"
    )

    if "corrected_speed_mps" not in speed.columns:
        raise ValueError(
            "imu_speed_correction.csv does not contain "
            "corrected_speed_mps."
        )

    speed["corrected_speed_mps"] = pd.to_numeric(
        speed["corrected_speed_mps"], errors="coerce"
    )

    if "speed_confidence" in speed.columns:
        speed["speed_confidence"] = pd.to_numeric(
            speed["speed_confidence"], errors="coerce"
        )

    # AI predictions are only available at 12648 timestamps.
    ai["timestamp_ms"] = pd.to_numeric(
        ai["timestamp_ms"], errors="coerce"
    )
    ai["ai_speed_mps"] = pd.to_numeric(
        ai["ai_speed_mps"], errors="coerce"
    )
    ai["speed_confidence"] = pd.to_numeric(
        ai["speed_confidence"], errors="coerce"
    )

    ai = ai.dropna(
        subset=["timestamp_ms", "ai_speed_mps"]
    ).copy()

    ai = ai.sort_values("timestamp_ms")

    # ---------------------------------------------------------
    # Align AI speed to IMU timeline
    #
    # AI predictions occur approximately every 1 second.
    # Between predictions we hold the latest AI prediction.
    # This avoids creating NaN speed values.
    # ---------------------------------------------------------

    print()
    print("Aligning speed to IMU timeline...")

    ai_for_merge = ai[
        [
            "timestamp_ms",
            "ai_speed_mps",
            "speed_confidence",
        ]
    ].copy()

    ai_for_merge = ai_for_merge.rename(
        columns={
            "ai_speed_mps": "ai_speed_aligned",
            "speed_confidence": "ai_confidence_aligned",
        }
    )

    merged = pd.merge_asof(
        imu.sort_values("timestamp_ms"),
        ai_for_merge.sort_values("timestamp_ms"),
        on="timestamp_ms",
        direction="backward",
    )

    # ---------------------------------------------------------
    # Align corrected speed by exact timestamp
    #
    # corrected_speed exists only on AI prediction rows.
    # Therefore it is not required on every IMU row.
    # ---------------------------------------------------------

    corrected_lookup = speed[
        ["timestamp_ms", "corrected_speed_mps"]
    ].drop_duplicates("timestamp_ms")

    merged = merged.merge(
        corrected_lookup,
        on="timestamp_ms",
        how="left",
    )

    # ---------------------------------------------------------
    # Select speed
    #
    # 1. corrected speed where available
    # 2. otherwise held AI speed
    #
    # This gives every IMU row a finite speed after the
    # first AI prediction.
    # ---------------------------------------------------------

    merged["speed_mps"] = merged["corrected_speed_mps"]

    corrected_mask = (
        merged["corrected_speed_mps"].notna()
        & np.isfinite(
            merged["corrected_speed_mps"].to_numpy(dtype=float)
        )
    )

    ai_mask = (
        ~corrected_mask
        & merged["ai_speed_aligned"].notna()
    )

    merged.loc[ai_mask, "speed_mps"] = merged.loc[
        ai_mask, "ai_speed_aligned"
    ]

    merged["speed_source"] = "AI_FALLBACK"

    merged.loc[corrected_mask, "speed_source"] = (
        "IMU_CORRECTED_AI"
    )

    merged["speed_confidence"] = merged[
        "ai_confidence_aligned"
    ]

    # ---------------------------------------------------------
    # Remove rows before first AI prediction
    #
    # There is no valid AI/corrected speed before timestamp
    # 8309 ms. Preserve the actual usable navigation timeline.
    # ---------------------------------------------------------

    first_valid = merged["speed_mps"].notna()

    if not first_valid.any():
        raise ValueError(
            "No valid speed could be aligned to the IMU timeline."
        )

    first_index = first_valid.idxmax()

    merged = merged.loc[first_index:].copy()

    # Fill confidence where corrected speed exists but AI
    # confidence is unavailable.
    merged["speed_confidence"] = merged[
        "speed_confidence"
    ].fillna(0.0)

    # ---------------------------------------------------------
    # Speed validation
    # ---------------------------------------------------------

    merged["speed_mps"] = pd.to_numeric(
        merged["speed_mps"], errors="coerce"
    )

    if merged["speed_mps"].isna().any():
        raise ValueError(
            "Speed alignment still contains missing values."
        )

    if not np.isfinite(
        merged["speed_mps"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Speed contains NaN or Inf."
        )

    if (merged["speed_mps"] < 0).any():
        raise ValueError(
            "Speed contains negative values."
        )

    # ---------------------------------------------------------
    # Build input for existing INS implementation
    #
    # ins.propagate_2d expects:
    # timestamp_ms
    # ai_speed_mps
    # heading_deg
    # yaw_rate
    #
    # We deliberately reuse the existing implementation.
    # ---------------------------------------------------------

    ins_input = pd.DataFrame()

    ins_input["timestamp_ms"] = merged["timestamp_ms"]

    ins_input["ai_speed_mps"] = merged["speed_mps"]

    # Existing ins.py applies +180 degrees internally.
    # Therefore provide the equivalent phone azimuth:
    #
    # travel_heading = azimuth + 180
    #
    # => azimuth = travel_heading - 180
    ins_input["heading_deg"] = (
        merged["travel_heading_deg"] - 180.0
    ) % 360.0

    # yaw_rate is diagnostic only.
    if "gyro_forward_mps2" in merged.columns:
        ins_input["yaw_rate"] = 0.0
    else:
        ins_input["yaw_rate"] = 0.0

    # ---------------------------------------------------------
    # Run existing 2D propagation
    # ---------------------------------------------------------

    print()
    print("Running existing ins.propagate_2d...")
    print("Method: speed + IMU heading")
    print("Raw acceleration is NOT integrated.")

    trajectory = propagate_2d(ins_input)

    # ---------------------------------------------------------
    # Build final output
    # ---------------------------------------------------------

    output = pd.DataFrame()

    output["timestamp_ms"] = merged["timestamp_ms"].to_numpy()

    output["dt_s"] = trajectory["dt_s"].to_numpy()

    output["speed_mps"] = merged["speed_mps"].to_numpy()

    output["speed_source"] = merged[
        "speed_source"
    ].to_numpy()

    output["speed_confidence"] = merged[
        "speed_confidence"
    ].to_numpy()

    output["travel_heading_deg"] = (
        merged["travel_heading_deg"].to_numpy()
    )

    output["accel_forward_mps2"] = merged[
        "accel_forward_mps2"
    ].to_numpy()

    output["accel_lateral_mps2"] = merged[
        "accel_lateral_mps2"
    ].to_numpy()

    output["accel_east_mps2"] = merged[
        "accel_east_mps2"
    ].to_numpy()

    output["accel_north_mps2"] = merged[
        "accel_north_mps2"
    ].to_numpy()

    # Existing propagate_2d uses x=east, y=north.
    output["velocity_east_mps"] = (
        output["speed_mps"]
        * np.sin(
            np.deg2rad(output["travel_heading_deg"])
        )
    )

    output["velocity_north_mps"] = (
        output["speed_mps"]
        * np.cos(
            np.deg2rad(output["travel_heading_deg"])
        )
    )

    output["x_m"] = trajectory["x_m"].to_numpy()
    output["y_m"] = trajectory["y_m"].to_numpy()

    # ---------------------------------------------------------
    # Convert local ENU position to latitude/longitude
    # ---------------------------------------------------------

    origin_lat = 52.047493
    origin_lon = -0.756155

    latitudes = []
    longitudes = []

    for x, y in zip(
        output["x_m"],
        output["y_m"],
    ):
        lat, lon = xy_to_latlon(
            float(x),
            float(y),
            origin_lat,
            origin_lon,
        )

        latitudes.append(lat)
        longitudes.append(lon)

    output["latitude"] = latitudes
    output["longitude"] = longitudes

    output["mode"] = "IMU_DEAD_RECKONING"

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("VALIDATION")
    print("-" * 60)

    expected_rows = len(imu)
    actual_rows = len(output)

    print(
        f"Original IMU rows          : {expected_rows}"
    )

    print(
        f"Rows before first AI       : "
        f"{expected_rows - actual_rows}"
    )

    print(
        f"Output rows                : {actual_rows}"
    )

    print(
        f"Timestamp order            : "
        f"{output['timestamp_ms'].is_monotonic_increasing}"
    )

    print(
        f"Timestamp duplicates       : "
        f"{output['timestamp_ms'].duplicated().sum()}"
    )

    numeric_columns = [
        "timestamp_ms",
        "dt_s",
        "speed_mps",
        "speed_confidence",
        "travel_heading_deg",
        "accel_forward_mps2",
        "accel_lateral_mps2",
        "accel_east_mps2",
        "accel_north_mps2",
        "velocity_east_mps",
        "velocity_north_mps",
        "x_m",
        "y_m",
        "latitude",
        "longitude",
    ]

    finite_check(
        output,
        numeric_columns,
        "Output",
    )

    print("NaN/Inf                   : PASS")

    print(
        f"Speed min/max              : "
        f"{output['speed_mps'].min():.4f} / "
        f"{output['speed_mps'].max():.4f} m/s"
    )

    print(
        f"Negative speed             : "
        f"{(output['speed_mps'] < 0).sum()}"
    )

    heading_ok = (
        output["travel_heading_deg"].between(
            0.0,
            360.0,
            inclusive="left",
        ).all()
    )

    print(
        f"Heading [0,360)            : "
        f"{heading_ok}"
    )

    print()
    print("SPEED SOURCES:")
    print(
        output["speed_source"].value_counts().to_string()
    )

    final_x = float(output["x_m"].iloc[-1])
    final_y = float(output["y_m"].iloc[-1])

    net_displacement = math.sqrt(
        final_x ** 2 + final_y ** 2
    )

    path_length = (
        np.sqrt(
            np.diff(output["x_m"]) ** 2
            + np.diff(output["y_m"]) ** 2
        ).sum()
    )

    print()
    print(
        f"Final x (east)            : "
        f"{final_x:.2f} m"
    )

    print(
        f"Final y (north)           : "
        f"{final_y:.2f} m"
    )

    print(
        f"Net displacement          : "
        f"{net_displacement:.2f} m"
    )

    print(
        f"Integrated path length    : "
        f"{path_length:.2f} m"
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("VALIDATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()