"""IMU + AI speed + GNSS EKF for the S-Vw4 experiment.

This version is deliberately self-contained and robust to:
- the S-Vw4 source directory layout,
- AI predictions being available only at ~1 Hz,
- the IMU correction file containing NaN AI fields before the first prediction,
- duplicate AI columns created by pandas merges,
- the 200-sample / 19.899 s GNSS blackout,
- the calibrated phone-heading convention.

State:
    [east_position_m, north_position_m, east_velocity_mps, north_velocity_mps]

During GNSS blackout:
    - GNSS position updates are disabled.
    - AI speed remains the velocity constraint.
    - phone heading + 180 deg + calibrated bias gives travel direction.
    - IMU acceleration is attenuated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ekf import PositionVelocityEKF


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
)

SVW4_DIR = DATA_ROOT / "S-Dataset"

SOURCE_FILE = SVW4_DIR / "S-Vw4.csv"
NAVIGATION_FILE = PROJECT_ROOT / "results" / "navigation_gnss_blackout.csv"
IMU_CORRECTION_FILE = PROJECT_ROOT / "results" / "imu_speed_correction.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ekf_imu_corrected_trajectory.csv"


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

EXPECTED_BLACKOUT_ROWS = 200

# Calibrated from the blackout heading comparison.
# Raw phone heading -> travel heading:
#     phone + 180 + bias
HEADING_BIAS_DEG = 20.296

# EKF parameters.
# Keep these consistent with the validation script.
PROCESS_ACCEL_NOISE = 5.0
GNSS_POSITION_NOISE = 3.0
AI_SPEED_NOISE = 1.0
BLACKOUT_AI_SPEED_NOISE = 1.0

# Reduce the effect of noisy phone-frame acceleration during
# GNSS outage. AI speed is the primary velocity constraint.
BLACKOUT_IMU_ACCEL_SCALE = 0.10

# Alignment tolerance between the 10-Hz navigation stream and
# the 10-Hz IMU correction stream.
IMU_ALIGNMENT_TOLERANCE_MS = 60

# Alignment tolerance for source GNSS.
GNSS_ALIGNMENT_TOLERANCE_MS = 60


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing columns: {', '.join(missing)}"
        )


def numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    frame[column] = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    return frame[column]


def finite_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    frame[column] = values

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{name}: column '{column}' contains NaN or Inf."
        )


def finite_or_nan_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    frame[column] = values

    arr = values.to_numpy(dtype=float)
    bad = np.isinf(arr)

    if bad.any():
        raise ValueError(
            f"{name}: column '{column}' contains Inf."
        )


# ============================================================
# HEADING
# ============================================================

def corrected_travel_heading(
    heading_deg: float,
) -> tuple[float, float]:
    """Return corrected compass travel heading and math angle.

    S-Vw4 phone heading convention is:
        travel_heading = phone_heading + 180 + calibrated_bias

    Compass:
        0   = north
        90  = east
        180 = south
        270 = west

    Mathematical angle:
        0   = east
        pi/2 = north
    """

    if not math.isfinite(heading_deg):
        raise ValueError("Heading must be finite.")

    travel_heading = (
        float(heading_deg)
        + 180.0
        + HEADING_BIAS_DEG
    ) % 360.0

    theta = math.radians(
        90.0 - travel_heading
    )

    return travel_heading, theta


# ============================================================
# LOAD NAVIGATION
# ============================================================

def load_navigation() -> pd.DataFrame:
    if not NAVIGATION_FILE.exists():
        raise FileNotFoundError(
            f"Navigation file not found:\n{NAVIGATION_FILE}"
        )

    navigation = pd.read_csv(NAVIGATION_FILE)

    navigation.columns = (
        navigation.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "timestamp_ms",
        "ai_speed_mps",
        "speed_confidence",
        "heading_deg",
        "yaw_rate",
        "motion_state",
        "gnss_available",
    ]

    require_columns(
        navigation,
        required,
        "Navigation",
    )

    for column in [
        "timestamp_ms",
        "ai_speed_mps",
        "speed_confidence",
        "heading_deg",
        "yaw_rate",
    ]:
        numeric_column(
            navigation,
            column,
        )

    navigation["gnss_available"] = (
        navigation["gnss_available"]
        .astype(bool)
    )

    navigation = (
        navigation
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    finite_column(
        navigation,
        "timestamp_ms",
        "Navigation",
    )

    finite_column(
        navigation,
        "heading_deg",
        "Navigation",
    )

    finite_or_nan_column(
        navigation,
        "ai_speed_mps",
        "Navigation",
    )

    finite_or_nan_column(
        navigation,
        "speed_confidence",
        "Navigation",
    )

    if navigation["timestamp_ms"].duplicated().any():
        raise ValueError(
            "Navigation contains duplicate timestamps."
        )

    if not navigation["timestamp_ms"].is_monotonic_increasing:
        raise ValueError(
            "Navigation timestamps are not ordered."
        )

    return navigation


# ============================================================
# LOAD IMU CORRECTION
# ============================================================

def load_imu() -> pd.DataFrame:
    if not IMU_CORRECTION_FILE.exists():
        raise FileNotFoundError(
            f"IMU correction file not found:\n"
            f"{IMU_CORRECTION_FILE}"
        )

    imu = pd.read_csv(
        IMU_CORRECTION_FILE
    )

    imu.columns = (
        imu.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
        "corrected_speed_mps",
        "ai_speed_mps",
        "speed_confidence",
    ]

    require_columns(
        imu,
        required,
        "IMU correction",
    )

    # Physical IMU columns must be finite.
    finite_columns = [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
    ]

    for column in finite_columns:
        finite_column(
            imu,
            column,
            "IMU correction",
        )

    # AI-derived fields are allowed to be NaN because the IMU
    # stream begins before the first AI prediction.
    for column in [
        "ai_speed_mps",
        "speed_confidence",
        "corrected_speed_mps",
    ]:
        finite_or_nan_column(
            imu,
            column,
            "IMU correction",
        )

    imu = (
        imu
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    if imu["timestamp_ms"].duplicated().any():
        raise ValueError(
            "IMU correction contains duplicate timestamps."
        )

    if not imu["timestamp_ms"].is_monotonic_increasing:
        raise ValueError(
            "IMU correction timestamps are not ordered."
        )

    return imu


# ============================================================
# LOAD SOURCE GNSS
# ============================================================

def load_gnss() -> pd.DataFrame:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"GNSS source file not found:\n{SOURCE_FILE}"
        )

    source = pd.read_csv(
        SOURCE_FILE,
        encoding="latin1",
    )

    source.columns = (
        source.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "TIME SINCE START (ms)",
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
    ]

    require_columns(
        source,
        required,
        "S-Vw4 source",
    )

    gnss = source[
        required
    ].copy()

    gnss = gnss.rename(
        columns={
            "TIME SINCE START (ms)": "gps_timestamp_ms",
            "GPS LATITUDE (degrees)": "gps_latitude_deg",
            "GPS LONGITUDE (degrees)": "gps_longitude_deg",
        }
    )

    for column in [
        "gps_timestamp_ms",
        "gps_latitude_deg",
        "gps_longitude_deg",
    ]:
        numeric_column(
            gnss,
            column,
        )

    gnss = gnss.dropna()

    gnss = (
        gnss
        .sort_values("gps_timestamp_ms")
        .reset_index(drop=True)
    )

    if gnss.empty:
        raise ValueError(
            "S-Vw4 source contains no valid GNSS rows."
        )

    gnss = (
        gnss
        .drop_duplicates(
            "gps_timestamp_ms",
            keep="first",
        )
        .reset_index(drop=True)
    )

    return gnss


# ============================================================
# GPS -> LOCAL EAST/NORTH
# ============================================================

def gps_to_local_xy(
    latitude_deg: float,
    longitude_deg: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[float, float]:
    lat_rad = math.radians(
        origin_lat_deg
    )

    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = (
        111320.0
        * math.cos(lat_rad)
    )

    x = (
        longitude_deg
        - origin_lon_deg
    ) * meters_per_degree_lon

    y = (
        latitude_deg
        - origin_lat_deg
    ) * meters_per_degree_lat

    return float(x), float(y)


# ============================================================
# ATTACH GNSS
# ============================================================

def attach_gnss(
    navigation: pd.DataFrame,
    gnss: pd.DataFrame,
) -> pd.DataFrame:
    nav = (
        navigation
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
        .copy()
    )

    gps = (
        gnss
        .sort_values("gps_timestamp_ms")
        .reset_index(drop=True)
        .copy()
    )

    merged = pd.merge_asof(
        nav,
        gps,
        left_on="timestamp_ms",
        right_on="gps_timestamp_ms",
        direction="nearest",
        tolerance=GNSS_ALIGNMENT_TOLERANCE_MS,
    )

    valid = (
        merged["gps_latitude_deg"].notna()
        & merged["gps_longitude_deg"].notna()
    )

    if not valid.any():
        raise ValueError(
            "No GNSS samples aligned to navigation timeline."
        )

    origin = merged.loc[
        valid
    ].iloc[0]

    origin_lat = float(
        origin["gps_latitude_deg"]
    )
    origin_lon = float(
        origin["gps_longitude_deg"]
    )

    x_values: list[float] = []
    y_values: list[float] = []

    for lat, lon in zip(
        merged["gps_latitude_deg"],
        merged["gps_longitude_deg"],
    ):
        if pd.isna(lat) or pd.isna(lon):
            x_values.append(np.nan)
            y_values.append(np.nan)
            continue

        x, y = gps_to_local_xy(
            float(lat),
            float(lon),
            origin_lat,
            origin_lon,
        )

        x_values.append(x)
        y_values.append(y)

    merged["gnss_x_m"] = x_values
    merged["gnss_y_m"] = y_values

    return merged


# ============================================================
# ATTACH IMU
# ============================================================

def attach_imu(
    navigation: pd.DataFrame,
    imu: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only IMU-specific fields.

    Navigation remains authoritative for:
        ai_speed_mps
        speed_confidence
        heading_deg
        yaw_rate
        motion_state
        gnss_available

    This avoids pandas creating ai_speed_mps_x / ai_speed_mps_y
    and eliminates the KeyError seen in the previous version.
    """

    nav = (
        navigation
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
        .copy()
    )

    imu_columns = [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
        "corrected_speed_mps",
    ]

    imu_for_merge = (
        imu[imu_columns]
        .copy()
        .rename(
            columns={
                "timestamp_ms": "imu_timestamp_ms",
                "linear_ax": "imu_linear_ax",
                "linear_ay": "imu_linear_ay",
                "corrected_speed_mps":
                    "imu_corrected_speed_mps",
            }
        )
        .sort_values("imu_timestamp_ms")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        nav,
        imu_for_merge,
        left_on="timestamp_ms",
        right_on="imu_timestamp_ms",
        direction="nearest",
        tolerance=IMU_ALIGNMENT_TOLERANCE_MS,
    )

    return merged


# ============================================================
# SPEED SELECTION
# ============================================================

def choose_speed(
    row: pd.Series,
) -> tuple[float | None, str]:
    """Choose speed without ever turning NaN into zero.

    Priority:
        1. corrected AI speed
        2. raw AI speed
        3. none
    """

    corrected = row.get(
        "imu_corrected_speed_mps",
        np.nan,
    )

    ai_speed = row.get(
        "ai_speed_mps",
        np.nan,
    )

    if (
        pd.notna(corrected)
        and math.isfinite(float(corrected))
        and float(corrected) >= 0.0
    ):
        return (
            float(corrected),
            "AI_CORRECTED",
        )

    if (
        pd.notna(ai_speed)
        and math.isfinite(float(ai_speed))
        and float(ai_speed) >= 0.0
    ):
        return (
            float(ai_speed),
            "AI_FALLBACK",
        )

    return (
        None,
        "IMU_ONLY",
    )


def is_actual_ai_prediction(
    row: pd.Series,
) -> bool:
    value = row.get(
        "ai_speed_mps",
        np.nan,
    )

    return (
        pd.notna(value)
        and math.isfinite(float(value))
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("IMU + AI SPEED + GNSS EKF")
    print("=" * 60)

    navigation = load_navigation()
    imu = load_imu()
    gnss = load_gnss()

    # Attach IMU without duplicating navigation AI columns.
    merged = attach_imu(
        navigation,
        imu,
    )

    # Attach source GNSS.
    merged = attach_gnss(
        merged,
        gnss,
    )

    # Navigation's blackout flag is authoritative.
    blackout_mask = (
        ~merged["gnss_available"].astype(bool)
    )

    blackout_rows = int(
        blackout_mask.sum()
    )

    if blackout_rows != EXPECTED_BLACKOUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_BLACKOUT_ROWS} "
            f"blackout rows, found {blackout_rows}"
        )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    first = merged.iloc[0]

    initial_heading_deg = float(
        first["heading_deg"]
    )

    initial_travel_heading_deg, _ = (
        corrected_travel_heading(
            initial_heading_deg
        )
    )

    initial_speed, _ = choose_speed(
        first
    )

    if initial_speed is None:
        initial_speed = 0.0

    initial_heading_rad = math.radians(
        initial_travel_heading_deg
    )

    initial_vx = (
        initial_speed
        * math.sin(initial_heading_rad)
    )

    initial_vy = (
        initial_speed
        * math.cos(initial_heading_rad)
    )

    first_gnss_x = first["gnss_x_m"]
    first_gnss_y = first["gnss_y_m"]

    if (
        pd.notna(first_gnss_x)
        and pd.notna(first_gnss_y)
    ):
        initial_x = float(
            first_gnss_x
        )
        initial_y = float(
            first_gnss_y
        )
    else:
        initial_x = 0.0
        initial_y = 0.0

    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    ekf = PositionVelocityEKF(
        initial_x=initial_x,
        initial_y=initial_y,
        initial_vx=initial_vx,
        initial_vy=initial_vy,
        process_accel_noise=PROCESS_ACCEL_NOISE,
        gnss_position_noise=GNSS_POSITION_NOISE,
        speed_noise=AI_SPEED_NOISE,
    )

    records: list[dict] = []

    previous_timestamp: int | None = None

    ai_updates = 0
    gnss_updates = 0
    corrected_speed_rows = 0
    fallback_speed_rows = 0

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for _, row in merged.iterrows():
        timestamp = int(
            row["timestamp_ms"]
        )

        if previous_timestamp is None:
            dt = 0.0
        else:
            dt = (
                timestamp
                - previous_timestamp
            ) / 1000.0

        previous_timestamp = timestamp

        if dt < 0.0:
            raise ValueError(
                "Navigation timestamps are not increasing."
            )

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        heading_deg = float(
            row["heading_deg"]
        )

        travel_heading_deg, theta = (
            corrected_travel_heading(
                heading_deg
            )
        )

        # ----------------------------------------------------
        # IMU ACCELERATION
        # ----------------------------------------------------

        ax_body = row.get(
            "imu_linear_ax",
            np.nan,
        )
        ay_body = row.get(
            "imu_linear_ay",
            np.nan,
        )

        if pd.isna(ax_body):
            ax_body = 0.0

        if pd.isna(ay_body):
            ay_body = 0.0

        ax_body = float(ax_body)
        ay_body = float(ay_body)

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        ax_east = (
            ax_body * cos_theta
            - ay_body * sin_theta
        )

        ay_north = (
            ax_body * sin_theta
            + ay_body * cos_theta
        )

        gnss_available = bool(
            row["gnss_available"]
        )

        if not gnss_available:
            ax_east *= BLACKOUT_IMU_ACCEL_SCALE
            ay_north *= BLACKOUT_IMU_ACCEL_SCALE

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        if dt > 0.0:
            ekf.predict(
                ax_east,
                ay_north,
                dt,
            )

        # ----------------------------------------------------
        # AI SPEED UPDATE
        # ----------------------------------------------------

        has_ai = is_actual_ai_prediction(
            row
        )

        selected_speed, speed_source = (
            choose_speed(row)
        )

        if has_ai and selected_speed is not None:
            if speed_source == "AI_CORRECTED":
                corrected_speed_rows += 1
            elif speed_source == "AI_FALLBACK":
                fallback_speed_rows += 1

            speed_noise = (
                BLACKOUT_AI_SPEED_NOISE
                if not gnss_available
                else AI_SPEED_NOISE
            )

            ekf.update_speed(
                selected_speed,
                math.radians(
                    travel_heading_deg
                ),
                noise=speed_noise,
            )

            ai_updates += 1
        else:
            speed_source = "IMU_ONLY"

        # ----------------------------------------------------
        # GNSS UPDATE
        #
        # Never update with GNSS during the blackout.
        # ----------------------------------------------------

        used_gnss = False

        if gnss_available:
            gx = row["gnss_x_m"]
            gy = row["gnss_y_m"]

            if (
                pd.notna(gx)
                and pd.notna(gy)
            ):
                ekf.update_gnss(
                    float(gx),
                    float(gy),
                )

                used_gnss = True
                gnss_updates += 1

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        x, y, vx, vy = ekf.state()

        ekf_speed = math.hypot(
            vx,
            vy,
        )

        ai_speed_value = row.get(
            "ai_speed_mps",
            np.nan,
        )

        corrected_speed_value = row.get(
            "imu_corrected_speed_mps",
            np.nan,
        )

        if (
            pd.notna(ai_speed_value)
            and math.isfinite(float(ai_speed_value))
        ):
            ai_speed_output = float(
                ai_speed_value
            )
        else:
            ai_speed_output = np.nan

        if (
            pd.notna(corrected_speed_value)
            and math.isfinite(float(corrected_speed_value))
        ):
            corrected_speed_output = float(
                corrected_speed_value
            )
        else:
            corrected_speed_output = np.nan

        speed_error = (
            ekf_speed - selected_speed
            if selected_speed is not None
            else np.nan
        )

        measurement_noise_r = (
            BLACKOUT_AI_SPEED_NOISE
            if (
                not gnss_available
                and has_ai
            )
            else AI_SPEED_NOISE
        )

        records.append(
            {
                "timestamp_ms": timestamp,
                "x_m": x,
                "y_m": y,
                "vx_mps": vx,
                "vy_mps": vy,
                "ekf_speed_mps": ekf_speed,
                "heading_deg": heading_deg,
                "travel_heading_deg":
                    travel_heading_deg,
                "gnss_available":
                    gnss_available,
                "dt_s": dt,
                "mode": (
                    "GNSS_INS"
                    if used_gnss
                    else "DEAD_RECKONING"
                ),
                "speed_source":
                    speed_source,
                "ai_speed_mps":
                    ai_speed_output,
                "corrected_speed_mps":
                    corrected_speed_output,
                "imu_accel_east_mps2":
                    ax_east,
                "imu_accel_north_mps2":
                    ay_north,
                "selected_speed_mps":
                    (
                        selected_speed
                        if selected_speed is not None
                        else np.nan
                    ),
                "speed_error_mps":
                    speed_error,
                "process_noise_q":
                    PROCESS_ACCEL_NOISE,
                "measurement_noise_r":
                    measurement_noise_r,
            }
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = pd.DataFrame(records)

    if len(output) != len(navigation):
        raise ValueError(
            "EKF output row count does not match "
            "navigation row count."
        )

    if output["timestamp_ms"].duplicated().any():
        raise ValueError(
            "EKF output contains duplicate timestamps."
        )

    if not output[
        "timestamp_ms"
    ].is_monotonic_increasing:
        raise ValueError(
            "EKF output timestamps are not ordered."
        )

    # Actual EKF state must always be finite.
    state_columns = [
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "ekf_speed_mps",
        "travel_heading_deg",
        "dt_s",
        "imu_accel_east_mps2",
        "imu_accel_north_mps2",
    ]

    state_array = output[
        state_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        state_array
    ).all():
        raise FloatingPointError(
            "EKF state output contains NaN or Inf."
        )

    blackout_output = output[
        ~output["gnss_available"]
    ].copy()

    if len(blackout_output) != EXPECTED_BLACKOUT_ROWS:
        raise ValueError(
            "Final output blackout row count is incorrect."
        )

    # GNSS must be completely disabled during blackout.
    gnss_disabled_during_blackout = (
        not blackout_output[
            "gnss_available"
        ].any()
    )

    if not gnss_disabled_during_blackout:
        raise ValueError(
            "GNSS was used during blackout."
        )

    # AI speed must not disappear because of the merge.
    ai_update_rows = output[
        output["ai_speed_mps"].notna()
    ]

    if len(ai_update_rows) != 12648:
        raise ValueError(
            "Expected 12648 actual AI speed rows, "
            f"found {len(ai_update_rows)}."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    final_blackout = blackout_output.iloc[-1]

    print()
    print("=" * 60)
    print("EKF COMPLETE")
    print("=" * 60)

    print(
        f"Output rows        : {len(output)}"
    )
    print(
        f"Blackout rows      : {len(blackout_output)}"
    )
    print(
        f"IMU rows used      : {len(imu)}"
    )
    print(
        f"Corrected-speed rows: "
        f"{corrected_speed_rows}"
    )
    print(
        f"Fallback-speed rows : "
        f"{fallback_speed_rows}"
    )
    print(
        f"AI velocity updates : "
        f"{ai_updates}"
    )
    print(
        f"GNSS updates        : "
        f"{gnss_updates}"
    )
    print(
        "GNSS disabled during blackout: "
        f"{gnss_disabled_during_blackout}"
    )
    print(
        f"Heading bias applied: "
        f"{HEADING_BIAS_DEG:.3f} deg"
    )
    print(
        f"Final blackout x/y  : "
        f"{final_blackout['x_m']:.4f}, "
        f"{final_blackout['y_m']:.4f}"
    )
    print(
        f"Final blackout vx/vy: "
        f"{final_blackout['vx_mps']:.4f}, "
        f"{final_blackout['vy_mps']:.4f}"
    )
    print(
        f"Final blackout speed: "
        f"{final_blackout['ekf_speed_mps']:.4f}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
