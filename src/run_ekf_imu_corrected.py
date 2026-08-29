"""
IMU + AI SPEED + GNSS EKF
S-Vw4 production runner

IMPORTANT DATA MODEL
--------------------
navigation_gnss_blackout.csv:
    126477 rows at approximately 10 Hz.
    Contains:
        timestamp_ms
        ai_speed_mps
        speed_confidence
        heading_deg
        yaw_rate
        motion_state
        gnss_available

The navigation AI columns may be carried/repeated across the
10-Hz navigation timeline. Therefore:

    DO NOT use navigation["ai_speed_mps"].notna().sum()
    to determine the number of actual AI predictions.

The actual AI prediction timestamps are obtained from the
IMU correction stream, where the AI-derived fields are present
only on the approximately 1-Hz prediction timestamps.

State:
    [east_position_m,
     north_position_m,
     east_velocity_mps,
     north_velocity_mps]

During GNSS blackout:
    - GNSS position updates are completely disabled.
    - AI speed is applied only at actual AI prediction timestamps.
    - Phone heading determines velocity direction.
    - Travel heading = phone heading + 180 + calibrated bias.
    - IMU acceleration is strongly attenuated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ekf import PositionVelocityEKF
except ImportError:
    from src.ekf import PositionVelocityEKF


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

SOURCE_FILE = (
    DATA_ROOT
    / "S-Dataset"
    / "S-Vw4.csv"
)

NAVIGATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gnss_blackout.csv"
)

IMU_CORRECTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "imu_speed_correction.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ekf_imu_corrected_trajectory.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_BLACKOUT_ROWS = 200

EXPECTED_AI_ROWS = 12648

# Calibrated from the blackout heading comparison.
#
# Raw phone heading:
#     147.096 deg
#
# + 180:
#     327.096 deg
#
# + measured bias:
#     approximately 347.392 deg
#
HEADING_BIAS_DEG = 20.296


# EKF parameters.
#
# These match the validated pipeline configuration.
#
PROCESS_ACCEL_NOISE = 5.0
GNSS_POSITION_NOISE = 3.0

AI_SPEED_NOISE = 1.0
BLACKOUT_AI_SPEED_NOISE = 1.0

# During blackout the IMU acceleration is intentionally
# attenuated. AI speed + heading is the main velocity constraint.
BLACKOUT_IMU_ACCEL_SCALE = 0.10

# Alignment tolerances.
IMU_ALIGNMENT_TOLERANCE_MS = 60
GNSS_ALIGNMENT_TOLERANCE_MS = 60


# ============================================================
# HELPERS
# ============================================================

def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: "
            f"{', '.join(missing)}"
        )


def numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> None:
    frame[column] = pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def finite_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    numeric_column(
        frame,
        column,
    )

    values = frame[column].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}: column '{column}' "
            f"contains NaN or Inf."
        )


def finite_or_nan_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    numeric_column(
        frame,
        column,
    )

    values = frame[column].to_numpy(
        dtype=float
    )

    if np.isinf(values).any():
        raise ValueError(
            f"{name}: column '{column}' "
            f"contains Inf."
        )


def wrap_angle_deg(
    angle: float,
) -> float:
    return (
        float(angle)
        + 180.0
    ) % 360.0 - 180.0


# ============================================================
# HEADING
# ============================================================

def corrected_travel_heading(
    heading_deg: float,
) -> tuple[float, float]:
    """
    Convert phone heading to travel heading.

    S-Vw4 convention:

        travel_heading =
            phone_heading
            + 180 degrees
            + calibrated bias

    Compass:
        0   = North
        90  = East
        180 = South
        270 = West

    theta:
        mathematical angle
        0     = East
        pi/2  = North
    """

    heading_deg = float(
        heading_deg
    )

    if not math.isfinite(
        heading_deg
    ):
        raise ValueError(
            "Heading must be finite."
        )

    travel_heading = (
        heading_deg
        + 180.0
        + HEADING_BIAS_DEG
    ) % 360.0

    theta = math.radians(
        90.0 - travel_heading
    )

    return (
        float(travel_heading),
        float(theta),
    )


# ============================================================
# LOAD NAVIGATION
# ============================================================

def load_navigation() -> pd.DataFrame:

    if not NAVIGATION_FILE.exists():
        raise FileNotFoundError(
            "Navigation file not found:\n"
            f"{NAVIGATION_FILE}"
        )

    navigation = pd.read_csv(
        NAVIGATION_FILE
    )

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

    # IMPORTANT:
    # Do not force AI columns to be finite.
    # The navigation file is a 10-Hz timeline and may contain
    # carried/repeated AI values.
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

    finite_or_nan_column(
        navigation,
        "yaw_rate",
        "Navigation",
    )

    # Avoid bool("False") == True.
    if navigation[
        "gnss_available"
    ].dtype == object:

        navigation[
            "gnss_available"
        ] = (
            navigation[
                "gnss_available"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "1": True,
                    "yes": True,
                    "y": True,
                    "false": False,
                    "0": False,
                    "no": False,
                    "n": False,
                }
            )
        )

    if navigation[
        "gnss_available"
    ].isna().any():

        raise ValueError(
            "Navigation contains invalid "
            "gnss_available values."
        )

    navigation[
        "gnss_available"
    ] = navigation[
        "gnss_available"
    ].astype(bool)

    navigation = (
        navigation
        .sort_values(
            "timestamp_ms"
        )
        .reset_index(
            drop=True
        )
    )

    if navigation[
        "timestamp_ms"
    ].duplicated().any():

        raise ValueError(
            "Navigation contains "
            "duplicate timestamps."
        )

    if not navigation[
        "timestamp_ms"
    ].is_monotonic_increasing:

        raise ValueError(
            "Navigation timestamps are "
            "not ordered."
        )

    return navigation


# ============================================================
# LOAD IMU CORRECTION
# ============================================================

def load_imu() -> pd.DataFrame:

    if not IMU_CORRECTION_FILE.exists():
        raise FileNotFoundError(
            "IMU correction file not found:\n"
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

    # Physical IMU values must be finite.
    for column in [
        "timestamp_ms",
        "linear_ax",
        "linear_ay",
    ]:
        finite_column(
            imu,
            column,
            "IMU correction",
        )

    # AI-derived fields are allowed to be NaN.
    for column in [
        "corrected_speed_mps",
        "ai_speed_mps",
        "speed_confidence",
    ]:
        finite_or_nan_column(
            imu,
            column,
            "IMU correction",
        )

    imu = (
        imu
        .sort_values(
            "timestamp_ms"
        )
        .reset_index(
            drop=True
        )
    )

    if imu[
        "timestamp_ms"
    ].duplicated().any():

        raise ValueError(
            "IMU correction contains "
            "duplicate timestamps."
        )

    if not imu[
        "timestamp_ms"
    ].is_monotonic_increasing:

        raise ValueError(
            "IMU correction timestamps "
            "are not ordered."
        )

    return imu


# ============================================================
# ACTUAL AI PREDICTION STREAM
# ============================================================

def extract_actual_ai_stream(
    imu: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the actual approximately-1-Hz AI predictions.

    IMPORTANT:
    The navigation stream has 126477 rows.

    The actual AI stream has 12648 rows.

    We identify actual AI predictions from the IMU correction
    stream rather than from the repeated navigation AI field.
    """

    ai = imu[
        [
            "timestamp_ms",
            "ai_speed_mps",
            "speed_confidence",
            "corrected_speed_mps",
        ]
    ].copy()

    valid = (
        ai[
            "ai_speed_mps"
        ].notna()
        & np.isfinite(
            ai[
                "ai_speed_mps"
            ].to_numpy(
                dtype=float
            )
        )
    )

    ai = ai.loc[
        valid
    ].copy()

    ai = (
        ai
        .sort_values(
            "timestamp_ms"
        )
        .drop_duplicates(
            subset=[
                "timestamp_ms"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if len(ai) != EXPECTED_AI_ROWS:

        raise ValueError(
            "Actual AI prediction count "
            "is wrong.\n"
            f"Expected: {EXPECTED_AI_ROWS}\n"
            f"Found:    {len(ai)}\n\n"
            "The AI stream must be extracted "
            "from actual prediction timestamps, "
            "not from the 10-Hz navigation "
            "timeline."
        )

    return ai


# ============================================================
# LOAD SOURCE GNSS
# ============================================================

def load_gnss() -> pd.DataFrame:

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            "S-Vw4 source file not found:\n"
            f"{SOURCE_FILE}"
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
            "TIME SINCE START (ms)":
                "gps_timestamp_ms",

            "GPS LATITUDE (degrees)":
                "gps_latitude_deg",

            "GPS LONGITUDE (degrees)":
                "gps_longitude_deg",
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

    gnss = gnss.dropna(
        subset=[
            "gps_timestamp_ms",
            "gps_latitude_deg",
            "gps_longitude_deg",
        ]
    )

    gnss = (
        gnss
        .sort_values(
            "gps_timestamp_ms"
        )
        .drop_duplicates(
            subset=[
                "gps_timestamp_ms"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if gnss.empty:
        raise ValueError(
            "S-Vw4 source contains "
            "no valid GNSS rows."
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

    return (
        float(x),
        float(y),
    )


# ============================================================
# ATTACH GNSS
# ============================================================

def attach_gnss(
    navigation: pd.DataFrame,
    gnss: pd.DataFrame,
) -> pd.DataFrame:

    nav = (
        navigation
        .sort_values(
            "timestamp_ms"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    gps = (
        gnss
        .sort_values(
            "gps_timestamp_ms"
        )
        .reset_index(
            drop=True
        )
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
        merged[
            "gps_latitude_deg"
        ].notna()
        & merged[
            "gps_longitude_deg"
        ].notna()
    )

    if not valid.any():
        raise ValueError(
            "No GNSS samples aligned "
            "to navigation timeline."
        )

    origin = merged.loc[
        valid
    ].iloc[0]

    origin_lat = float(
        origin[
            "gps_latitude_deg"
        ]
    )

    origin_lon = float(
        origin[
            "gps_longitude_deg"
        ]
    )

    x_values = []
    y_values = []

    for lat, lon in zip(
        merged[
            "gps_latitude_deg"
        ],
        merged[
            "gps_longitude_deg"
        ],
    ):

        if (
            pd.isna(lat)
            or pd.isna(lon)
        ):
            x_values.append(
                np.nan
            )
            y_values.append(
                np.nan
            )
            continue

        x, y = gps_to_local_xy(
            float(lat),
            float(lon),
            origin_lat,
            origin_lon,
        )

        x_values.append(x)
        y_values.append(y)

    merged[
        "gnss_x_m"
    ] = x_values

    merged[
        "gnss_y_m"
    ] = y_values

    return merged


# ============================================================
# ATTACH IMU
# ============================================================

def attach_imu(
    navigation: pd.DataFrame,
    imu: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach only physical IMU fields.

    DO NOT merge AI columns from IMU into navigation.
    The navigation timeline already has AI-named columns,
    and merging them would create _x/_y columns.

    Actual AI timestamps are handled separately.
    """

    nav = (
        navigation
        .sort_values(
            "timestamp_ms"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    imu_for_merge = imu[
        [
            "timestamp_ms",
            "linear_ax",
            "linear_ay",
            "corrected_speed_mps",
        ]
    ].copy()

    imu_for_merge = (
        imu_for_merge
        .rename(
            columns={
                "timestamp_ms":
                    "imu_timestamp_ms",

                "linear_ax":
                    "imu_linear_ax",

                "linear_ay":
                    "imu_linear_ay",

                "corrected_speed_mps":
                    "imu_corrected_speed_mps",
            }
        )
        .sort_values(
            "imu_timestamp_ms"
        )
        .reset_index(
            drop=True
        )
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
    ai_speed: float,
    corrected_speed: float,
) -> tuple[float | None, str]:

    if (
        pd.notna(corrected_speed)
        and math.isfinite(
            float(corrected_speed)
        )
        and float(corrected_speed) >= 0.0
    ):
        return (
            float(corrected_speed),
            "AI_CORRECTED",
        )

    if (
        pd.notna(ai_speed)
        and math.isfinite(
            float(ai_speed)
        )
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("IMU + AI SPEED + GNSS EKF")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    navigation = load_navigation()

    imu = load_imu()

    ai = extract_actual_ai_stream(
        imu
    )

    gnss = load_gnss()

    print()
    print(
        f"Navigation rows : "
        f"{len(navigation)}"
    )

    print(
        f"Actual AI rows  : "
        f"{len(ai)}"
    )

    print(
        f"Raw S-Vw4 rows  : "
        f"{len(imu)}"
    )

    # --------------------------------------------------------
    # BUILD TIMELINE
    # --------------------------------------------------------

    merged = attach_imu(
        navigation,
        imu,
    )

    merged = attach_gnss(
        merged,
        gnss,
    )

    # --------------------------------------------------------
    # CREATE EXACT AI TIMESTAMP LOOKUP
    # --------------------------------------------------------
    #
    # This is the critical fix.
    #
    # AI timestamps come from the actual AI stream,
    # not from navigation["ai_speed_mps"].notna().
    #

    ai_lookup = ai.set_index(
        "timestamp_ms"
    )

    ai_timestamps = set(
        int(value)
        for value in ai[
            "timestamp_ms"
        ].to_numpy(
            dtype=np.int64
        )
    )

    # --------------------------------------------------------
    # VERIFY ALL AI TIMESTAMPS ALIGN
    # --------------------------------------------------------

    navigation_timestamps = set(
        int(value)
        for value in merged[
            "timestamp_ms"
        ].to_numpy(
            dtype=np.int64
        )
    )

    missing_ai_timestamps = (
        ai_timestamps
        - navigation_timestamps
    )

    if missing_ai_timestamps:

        sample_missing = sorted(
            missing_ai_timestamps
        )[:10]

        raise ValueError(
            "Some actual AI timestamps "
            "do not exist on the navigation "
            "timeline.\n"
            f"Missing count: "
            f"{len(missing_ai_timestamps)}\n"
            f"Examples: "
            f"{sample_missing}"
        )

    # --------------------------------------------------------
    # BLACKOUT
    # --------------------------------------------------------

    blackout_mask = (
        ~merged[
            "gnss_available"
        ].astype(bool)
    )

    blackout_rows = int(
        blackout_mask.sum()
    )

    if blackout_rows != (
        EXPECTED_BLACKOUT_ROWS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_BLACKOUT_ROWS} "
            f"blackout rows, found "
            f"{blackout_rows}"
        )

    blackout_indices = np.flatnonzero(
        blackout_mask.to_numpy()
    )

    if len(blackout_indices) > 0:

        blackout_start_idx = (
            int(blackout_indices[0])
        )

        blackout_end_idx = (
            int(blackout_indices[-1])
        )

        blackout_start_time = int(
            merged.iloc[
                blackout_start_idx
            ][
                "timestamp_ms"
            ]
        )

        blackout_end_time = int(
            merged.iloc[
                blackout_end_idx
            ][
                "timestamp_ms"
            ]
        )

        blackout_duration = (
            blackout_end_time
            - blackout_start_time
        ) / 1000.0

    else:

        blackout_start_idx = None
        blackout_end_idx = None
        blackout_start_time = None
        blackout_end_time = None
        blackout_duration = 0.0

    print(
        f"Blackout rows   : "
        f"{blackout_rows}"
    )

    print(
        f"Blackout time   : "
        f"{blackout_duration:.3f} s"
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    first = merged.iloc[0]

    first_heading = float(
        first[
            "heading_deg"
        ]
    )

    initial_travel_heading, _ = (
        corrected_travel_heading(
            first_heading
        )
    )

    # The first navigation row is usually before the first
    # actual AI prediction. Use the nearest available actual
    # AI prediction only if it occurs at the first timestamp.
    #
    # Otherwise initialize at zero speed and let the first
    # actual AI update constrain velocity.
    first_timestamp = int(
        first[
            "timestamp_ms"
        ]
    )

    if first_timestamp in ai_lookup.index:

        first_ai = ai_lookup.loc[
            first_timestamp
        ]

        initial_speed, _ = choose_speed(
            float(
                first_ai[
                    "ai_speed_mps"
                ]
            ),
            float(
                first_ai[
                    "corrected_speed_mps"
                ]
            )
            if pd.notna(
                first_ai[
                    "corrected_speed_mps"
                ]
            )
            else np.nan,
        )

    else:

        initial_speed = 0.0

    if initial_speed is None:
        initial_speed = 0.0

    initial_heading_rad = math.radians(
        initial_travel_heading
    )

    initial_vx = (
        initial_speed
        * math.sin(
            initial_heading_rad
        )
    )

    initial_vy = (
        initial_speed
        * math.cos(
            initial_heading_rad
        )
    )

    first_gnss_x = first[
        "gnss_x_m"
    ]

    first_gnss_y = first[
        "gnss_y_m"
    ]

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

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    records = []

    previous_timestamp = None

    ai_updates = 0
    gnss_updates = 0

    corrected_speed_rows = 0
    fallback_speed_rows = 0

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for _, row in merged.iterrows():

        timestamp = int(
            row[
                "timestamp_ms"
            ]
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
                "Navigation timestamps "
                "are not increasing."
            )

        if dt > 1.0:
            raise ValueError(
                f"Invalid dt={dt:.6f} "
                f"at timestamp "
                f"{timestamp}"
            )

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        heading_deg = float(
            row[
                "heading_deg"
            ]
        )

        travel_heading_deg, theta = (
            corrected_travel_heading(
                heading_deg
            )
        )

        travel_heading_rad = math.radians(
            travel_heading_deg
        )

        # ----------------------------------------------------
        # IMU
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

        ax_body = float(
            ax_body
        )

        ay_body = float(
            ay_body
        )

        # Existing phone/body mapping.
        #
        # The original S-Vw4 implementation maps body
        # acceleration into the world frame using the corrected
        # travel heading.

        cos_theta = math.cos(
            theta
        )

        sin_theta = math.sin(
            theta
        )

        ax_east = (
            ax_body * cos_theta
            - ay_body * sin_theta
        )

        ay_north = (
            ax_body * sin_theta
            + ay_body * cos_theta
        )

        gnss_available = bool(
            row[
                "gnss_available"
            ]
        )

        # ----------------------------------------------------
        # BLACKOUT IMU ATTENUATION
        # ----------------------------------------------------

        if not gnss_available:

            ax_east *= (
                BLACKOUT_IMU_ACCEL_SCALE
            )

            ay_north *= (
                BLACKOUT_IMU_ACCEL_SCALE
            )

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
        # ACTUAL AI TIMESTAMP
        # ----------------------------------------------------
        #
        # CRITICAL:
        #
        # Do not use:
        #
        #     row["ai_speed_mps"] is not NaN
        #
        # because the navigation stream can carry/repeat AI
        # values.
        #
        # Instead:
        #
        #     timestamp in ai_timestamps
        #
        # ----------------------------------------------------

        is_ai_timestamp = (
            timestamp
            in ai_timestamps
        )

        selected_speed = None
        speed_source = "IMU_ONLY"

        ai_speed_output = np.nan
        corrected_speed_output = np.nan

        speed_confidence_output = np.nan

        if is_ai_timestamp:

            ai_row = ai_lookup.loc[
                timestamp
            ]

            ai_speed_value = (
                ai_row[
                    "ai_speed_mps"
                ]
            )

            corrected_speed_value = (
                ai_row[
                    "corrected_speed_mps"
                ]
            )

            confidence_value = (
                ai_row[
                    "speed_confidence"
                ]
            )

            if pd.notna(
                ai_speed_value
            ):

                ai_speed_output = float(
                    ai_speed_value
                )

            if pd.notna(
                corrected_speed_value
            ):

                corrected_speed_output = float(
                    corrected_speed_value
                )

            if pd.notna(
                confidence_value
            ):

                speed_confidence_output = float(
                    confidence_value
                )

            selected_speed, speed_source = (
                choose_speed(
                    ai_speed_output,
                    corrected_speed_output,
                )
            )

            if (
                selected_speed is not None
            ):

                if (
                    speed_source
                    == "AI_CORRECTED"
                ):

                    corrected_speed_rows += 1

                elif (
                    speed_source
                    == "AI_FALLBACK"
                ):

                    fallback_speed_rows += 1

                if gnss_available:

                    speed_noise = (
                        AI_SPEED_NOISE
                    )

                else:

                    speed_noise = (
                        BLACKOUT_AI_SPEED_NOISE
                    )

                ekf.update_speed(
                    selected_speed,
                    travel_heading_rad,
                    noise=speed_noise,
                )

                ai_updates += 1

        # ----------------------------------------------------
        # GNSS UPDATE
        # ----------------------------------------------------
        #
        # Navigation gnss_available is authoritative.
        #
        # During blackout:
        #
        #     NO GNSS UPDATE.
        #
        # ----------------------------------------------------

        used_gnss = False

        if gnss_available:

            gx = row[
                "gnss_x_m"
            ]

            gy = row[
                "gnss_y_m"
            ]

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
        # EKF STATE
        # ----------------------------------------------------

        x, y, vx, vy = (
            ekf.state()
        )

        x = float(x)
        y = float(y)
        vx = float(vx)
        vy = float(vy)

        ekf_speed = math.hypot(
            vx,
            vy,
        )

        if not math.isfinite(
            ekf_speed
        ):

            raise FloatingPointError(
                "EKF speed became "
                "NaN/Inf at timestamp "
                f"{timestamp}"
            )

        # ----------------------------------------------------
        # SPEED ERROR
        # ----------------------------------------------------

        if (
            selected_speed is not None
            and math.isfinite(
                float(selected_speed)
            )
        ):

            speed_error = (
                ekf_speed
                - float(selected_speed)
            )

        else:

            speed_error = np.nan

        # ----------------------------------------------------
        # MEASUREMENT NOISE
        # ----------------------------------------------------

        measurement_noise_r = (
            BLACKOUT_AI_SPEED_NOISE
            if (
                not gnss_available
                and is_ai_timestamp
                and selected_speed is not None
            )
            else AI_SPEED_NOISE
        )

        # ----------------------------------------------------
        # OUTPUT RECORD
        # ----------------------------------------------------

        records.append(
            {
                "timestamp_ms":
                    timestamp,

                "x_m":
                    x,

                "y_m":
                    y,

                "vx_mps":
                    vx,

                "vy_mps":
                    vy,

                "ekf_speed_mps":
                    ekf_speed,

                "heading_deg":
                    heading_deg,

                "travel_heading_deg":
                    travel_heading_deg,

                "gnss_available":
                    gnss_available,

                "dt_s":
                    dt,

                "mode":
                    (
                        "GNSS_INS"
                        if used_gnss
                        else "DEAD_RECKONING"
                    ),

                "speed_source":
                    speed_source,

                "ai_speed_mps":
                    (
                        ai_speed_output
                        if is_ai_timestamp
                        else np.nan
                    ),

                "speed_confidence":
                    (
                        speed_confidence_output
                        if is_ai_timestamp
                        else np.nan
                    ),

                "corrected_speed_mps":
                    (
                        corrected_speed_output
                        if is_ai_timestamp
                        else np.nan
                    ),

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

    output = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if len(output) != len(
        navigation
    ):

        raise ValueError(
            "EKF output row count does "
            "not match navigation row count."
        )

    if output[
        "timestamp_ms"
    ].duplicated().any():

        raise ValueError(
            "EKF output contains "
            "duplicate timestamps."
        )

    if not output[
        "timestamp_ms"
    ].is_monotonic_increasing:

        raise ValueError(
            "EKF output timestamps "
            "are not ordered."
        )

    # --------------------------------------------------------
    # STATE FINITENESS
    # --------------------------------------------------------

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
            "EKF state output contains "
            "NaN or Inf."
        )

    # --------------------------------------------------------
    # BLACKOUT VALIDATION
    # --------------------------------------------------------

    blackout_output = output[
        ~output[
            "gnss_available"
        ]
    ].copy()

    if len(
        blackout_output
    ) != EXPECTED_BLACKOUT_ROWS:

        raise ValueError(
            "Final output blackout "
            "row count is incorrect."
        )

    gnss_disabled_during_blackout = (
        not blackout_output[
            "gnss_available"
        ].any()
    )

    if not gnss_disabled_during_blackout:

        raise ValueError(
            "GNSS was used during blackout."
        )

    # --------------------------------------------------------
    # AI UPDATE VALIDATION
    # --------------------------------------------------------
    #
    # Count speed_source, not ai_speed_mps.notna().
    #
    # This must equal the actual 1-Hz prediction count.
    #

    ai_update_rows = output[
        output[
            "speed_source"
        ].isin(
            [
                "AI_CORRECTED",
                "AI_FALLBACK",
            ]
        )
    ]

    if len(
        ai_update_rows
    ) != EXPECTED_AI_ROWS:

        raise ValueError(
            "Actual AI update count is wrong.\n"
            f"Expected: {EXPECTED_AI_ROWS}\n"
            f"Found:    {len(ai_update_rows)}"
        )

    # --------------------------------------------------------
    # AI OUTPUT FINITENESS
    # --------------------------------------------------------

    ai_output = output[
        output[
            "speed_source"
        ].isin(
            [
                "AI_CORRECTED",
                "AI_FALLBACK",
            ]
        )
    ]

    for column in [
        "ai_speed_mps",
        "speed_confidence",
        "selected_speed_mps",
        "speed_error_mps",
    ]:

        if column == "speed_error_mps":

            # Speed error should be finite on every actual AI
            # update because the EKF state is finite.
            values = ai_output[
                column
            ].to_numpy(
                dtype=float
            )

        else:

            values = ai_output[
                column
            ].to_numpy(
                dtype=float
            )

        if not np.isfinite(
            values
        ).all():

            raise FloatingPointError(
                f"AI output column "
                f"'{column}' contains "
                f"NaN/Inf on an actual "
                f"AI update row."
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    final_blackout = (
        blackout_output.iloc[-1]
    )

    print()
    print("=" * 60)
    print("EKF COMPLETE")
    print("=" * 60)

    print(
        f"Output rows        : "
        f"{len(output)}"
    )

    print(
        f"Blackout rows      : "
        f"{len(blackout_output)}"
    )

    print(
        f"IMU rows used      : "
        f"{len(imu)}"
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
        f"Blackout IMU scale: "
        f"{BLACKOUT_IMU_ACCEL_SCALE:.2f}"
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