"""
S-Vw4 GPS reference evaluation and cleaning.

This module:
- loads S-Vw4 GPS data
- normalizes the source headers
- converts latitude/longitude to local x/y metres
- aligns GPS to EKF timestamps
- detects physically inconsistent GPS position jumps
- creates a cleaned GPS reference
- provides make_report() for evaluate_ekf_svw4.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PROJECT / FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

GPS_JUMP_THRESHOLD_M = 50.0

EARTH_RADIUS_M = 6378137.0


# ============================================================
# COLUMN HELPERS
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BOMs and leading/trailing whitespace from headers."""

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def find_column(
    df: pd.DataFrame,
    names: list[str],
    contains: Optional[list[str]] = None,
) -> str:
    """Find a column robustly."""

    columns = list(df.columns)

    # Exact.
    for name in names:
        if name in columns:
            return name

    # Case-insensitive.
    lower = {
        str(c).strip().lower(): c
        for c in columns
    }

    for name in names:
        key = name.strip().lower()

        if key in lower:
            return lower[key]

    # Contains.
    if contains:
        for column in columns:

            text = (
                str(column)
                .strip()
                .lower()
            )

            if all(
                token.lower() in text
                for token in contains
            ):
                return column

    raise KeyError(
        "Required column not found.\n"
        f"Requested: {names}\n"
        f"Available:\n"
        + "\n".join(
            repr(c)
            for c in columns
        )
    )


# ============================================================
# FIND S-Vw4
# ============================================================

def find_svw4_file() -> Path:
    """Find the S-Vw4 CSV in the project."""

    patterns = [
        "*S-Vw4*.csv",
        "*S-VW4*.csv",
        "*svw4*.csv",
        "*S_Vw4*.csv",
        "*S_VW4*.csv",
    ]

    ignored = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        "results",
        "node_modules",
    }

    candidates = []

    for pattern in patterns:

        for path in PROJECT_ROOT.rglob(pattern):

            if not path.is_file():
                continue

            if any(
                part in ignored
                for part in path.parts
            ):
                continue

            candidates.append(path)

    candidates = sorted(
        set(candidates),
        key=lambda p: (
            len(p.parts),
            str(p),
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find S-Vw4 CSV."
        )

    return candidates[0]


# ============================================================
# LOAD GPS
# ============================================================

def load_svw4_gps() -> pd.DataFrame:

    source_file = find_svw4_file()

    print(
        f"S-Vw4 reference : {source_file}"
    )

    print()
    print(
        "Loading S-Vw4 reference..."
    )

    source = pd.read_csv(
        source_file,
        encoding="latin1",
    )

    # CRITICAL:
    # source contains headers with leading spaces.
    source = normalize_columns(
        source
    )

    time_col = find_column(
        source,
        [
            "TIME SINCE START (ms)",
        ],
    )

    lat_col = find_column(
        source,
        [
            "GPS LATITUDE (degrees)",
        ],
    )

    lon_col = find_column(
        source,
        [
            "GPS LONGITUDE (degrees)",
        ],
    )

    speed_col = find_column(
        source,
        [
            "GPS SPEED (Kmh)",
        ],
    )

    accuracy_col = find_column(
        source,
        [
            "GPS ACCURACY (m)",
        ],
    )

    satellites_col = find_column(
        source,
        [
            "GPS SATELLITES IN RANGE",
        ],
    )

    print(
        f"Timestamp column : {time_col}"
    )

    print(
        f"Latitude column  : {lat_col}"
    )

    print(
        f"Longitude column : {lon_col}"
    )

    gps = pd.DataFrame()

    gps["timestamp_ms"] = pd.to_numeric(
        source[time_col],
        errors="coerce",
    )

    gps["latitude_deg"] = pd.to_numeric(
        source[lat_col],
        errors="coerce",
    )

    gps["longitude_deg"] = pd.to_numeric(
        source[lon_col],
        errors="coerce",
    )

    gps["reported_speed_kmh"] = pd.to_numeric(
        source[speed_col],
        errors="coerce",
    )

    gps["reported_speed_mps"] = (
        gps["reported_speed_kmh"] / 3.6
    )

    gps["gps_accuracy_m"] = pd.to_numeric(
        source[accuracy_col],
        errors="coerce",
    )

    gps["gps_satellites"] = (
        source[satellites_col]
        .astype(str)
    )

    gps = gps.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    gps = gps.dropna(
        subset=[
            "timestamp_ms",
            "latitude_deg",
            "longitude_deg",
        ]
    )

    gps = gps.sort_values(
        "timestamp_ms"
    )

    gps = gps.drop_duplicates(
        "timestamp_ms",
        keep="last",
    )

    gps = gps.reset_index(
        drop=True
    )

    if gps.empty:
        raise ValueError(
            "No valid S-Vw4 GPS rows."
        )

    # --------------------------------------------------------
    # LAT/LON -> LOCAL METRES
    # --------------------------------------------------------

    lat0 = np.radians(
        float(
            gps["latitude_deg"].iloc[0]
        )
    )

    lon0 = np.radians(
        float(
            gps["longitude_deg"].iloc[0]
        )
    )

    lat = np.radians(
        gps["latitude_deg"].to_numpy()
    )

    lon = np.radians(
        gps["longitude_deg"].to_numpy()
    )

    gps["x_m"] = (
        (lon - lon0)
        * np.cos(lat0)
        * EARTH_RADIUS_M
    )

    gps["y_m"] = (
        (lat - lat0)
        * EARTH_RADIUS_M
    )

    # --------------------------------------------------------
    # TIME DELTA
    # --------------------------------------------------------

    gps["gps_dt_s"] = (
        gps["timestamp_ms"].diff()
        / 1000.0
    )

    # --------------------------------------------------------
    # POSITION STEP
    # --------------------------------------------------------

    gps["gps_step_m"] = np.hypot(
        gps["x_m"].diff(),
        gps["y_m"].diff(),
    )

    gps.loc[
        gps.index[0],
        "gps_step_m",
    ] = 0.0

    # --------------------------------------------------------
    # POSITION-DERIVED SPEED
    # --------------------------------------------------------

    dt = gps["gps_dt_s"].clip(
        lower=0.05
    )

    gps["position_speed_mps"] = (
        gps["gps_step_m"] / dt
    )

    gps.loc[
        gps.index[0],
        "position_speed_mps",
    ] = 0.0

    # --------------------------------------------------------
    # POSITION SPEED / REPORTED SPEED
    # --------------------------------------------------------

    denominator = (
        gps["reported_speed_mps"]
        .abs()
        .clip(lower=0.5)
    )

    gps["speed_ratio"] = (
        gps["position_speed_mps"]
        / denominator
    )

    print(
        f"S-Vw4 GPS rows  : {len(gps)}"
    )

    return gps


# ============================================================
# GPS JUMP DETECTION
# ============================================================

def detect_gps_jumps(
    gps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect large GPS position jumps.

    This is diagnostic only.
    """

    return gps.loc[
        gps["gps_step_m"]
        > GPS_JUMP_THRESHOLD_M
    ].copy()


# ============================================================
# CLEAN GPS
# ============================================================

def clean_gps_reference(
    gps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    clean = gps.copy()

    clean["raw_x_m"] = (
        clean["x_m"]
    )

    clean["raw_y_m"] = (
        clean["y_m"]
    )

    # --------------------------------------------------------
    # IMPORTANT CLEANING RULE
    #
    # A GPS position is suspicious when:
    #
    # 1. its position jump exceeds 50 m, AND
    # 2. its implied speed is grossly inconsistent with the
    #    GPS-reported speed.
    #
    # This avoids blindly deleting every large displacement.
    # --------------------------------------------------------

    large_jump = (
        clean["gps_step_m"]
        > GPS_JUMP_THRESHOLD_M
    )

    position_speed = (
        clean["position_speed_mps"]
    )

    reported_speed = (
        clean["reported_speed_mps"]
        .abs()
        .fillna(0.0)
    )

    # Maximum physically plausible allowance.
    #
    # GPS can be noisy, so allow a generous factor.
    allowed_speed = np.maximum(
        reported_speed * 3.0,
        15.0,
    )

    speed_inconsistent = (
        position_speed
        > allowed_speed
    )

    # Also reject absurd instantaneous movement.
    absurd_speed = (
        position_speed
        > 40.0
    )

    clean["gps_jump"] = (
        large_jump
    )

    clean["speed_inconsistent"] = (
        speed_inconsistent
    )

    clean["gps_outlier"] = (
        large_jump
        & (
            speed_inconsistent
            | absurd_speed
        )
    )

    # First sample can never be an outlier.
    if len(clean) > 0:

        clean.loc[
            clean.index[0],
            "gps_outlier",
        ] = False

    # --------------------------------------------------------
    # INTERPOLATE ONLY OUTLIER POSITIONS
    # --------------------------------------------------------

    clean_x = clean["x_m"].where(
        ~clean["gps_outlier"],
        np.nan,
    )

    clean_y = clean["y_m"].where(
        ~clean["gps_outlier"],
        np.nan,
    )

    clean["x_m"] = (
        clean_x
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    clean["y_m"] = (
        clean_y
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    # --------------------------------------------------------
    # CLEANED STEP/SPEED
    # --------------------------------------------------------

    clean["clean_step_m"] = np.hypot(
        clean["x_m"].diff(),
        clean["y_m"].diff(),
    )

    clean.loc[
        clean.index[0],
        "clean_step_m",
    ] = 0.0

    clean["clean_position_speed_mps"] = (
        clean["clean_step_m"]
        / clean["gps_dt_s"].clip(
            lower=0.05
        )
    )

    clean.loc[
        clean.index[0],
        "clean_position_speed_mps",
    ] = 0.0

    jumps = clean.loc[
        clean["gps_jump"]
    ].copy()

    return clean, jumps


# ============================================================
# ALIGN GPS TO EKF
# ============================================================

def align_reference_to_ekf(
    ekf: pd.DataFrame,
    gps: pd.DataFrame,
) -> pd.DataFrame:

    left = (
        ekf[
            ["timestamp_ms"]
        ]
        .copy()
        .sort_values(
            "timestamp_ms"
        )
    )

    right = (
        gps[
            [
                "timestamp_ms",
                "x_m",
                "y_m",
            ]
        ]
        .copy()
        .sort_values(
            "timestamp_ms"
        )
    )

    right = right.rename(
        columns={
            "x_m": "gps_x_m",
            "y_m": "gps_y_m",
            "timestamp_ms":
                "gps_timestamp_ms",
        }
    )

    # Preserve original EKF timestamp.
    aligned = pd.merge_asof(
        left,
        right,
        left_on="timestamp_ms",
        right_on="gps_timestamp_ms",
        direction="nearest",
    )

    aligned[
        "gps_timestamp_difference_ms"
    ] = (
        aligned["timestamp_ms"]
        - aligned["gps_timestamp_ms"]
    ).abs()

    return aligned


# ============================================================
# ERROR CALCULATION
# ============================================================

def add_errors(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    dx = (
        df["x_m"]
        - df["gps_x_m"]
    )

    dy = (
        df["y_m"]
        - df["gps_y_m"]
    )

    df["position_error_x_m"] = dx

    df["position_error_y_m"] = dy

    df["position_error_m"] = np.hypot(
        dx,
        dy,
    )

    return df


# ============================================================
# BLACKOUT MASK
# ============================================================

def get_blackout_mask(
    df: pd.DataFrame,
) -> pd.Series:

    values = df[
        "gnss_available"
    ]

    if pd.api.types.is_bool_dtype(
        values
    ):
        return ~values.fillna(
            False
        )

    text = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    available = text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )

    return ~available


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df: pd.DataFrame,
    mask: pd.Series,
    period: str,
) -> dict:

    values = pd.to_numeric(
        df.loc[
            mask,
            "position_error_m",
        ],
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "period": period,
            "samples": 0,
            "mean_error_m": np.nan,
            "median_error_m": np.nan,
            "rmse_m": np.nan,
            "p95_error_m": np.nan,
            "max_error_m": np.nan,
        }

    values = values.to_numpy(
        dtype=float
    )

    return {
        "period": period,
        "samples": int(
            len(values)
        ),
        "mean_error_m": float(
            np.mean(values)
        ),
        "median_error_m": float(
            np.median(values)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        ),
        "p95_error_m": float(
            np.percentile(
                values,
                95,
            )
        ),
        "max_error_m": float(
            np.max(values)
        ),
    }


# ============================================================
# PUBLIC REPORT FUNCTION
# ============================================================

def make_report(
    df: pd.DataFrame,
) -> pd.DataFrame:

    blackout = get_blackout_mask(
        df
    )

    if blackout.any():

        start = df.loc[
            blackout,
            "timestamp_ms",
        ].min()

        end = df.loc[
            blackout,
            "timestamp_ms",
        ].max()

        before = (
            df["timestamp_ms"]
            < start
        )

        after = (
            df["timestamp_ms"]
            > end
        )

    else:

        before = pd.Series(
            True,
            index=df.index,
        )

        blackout = pd.Series(
            False,
            index=df.index,
        )

        after = pd.Series(
            False,
            index=df.index,
        )

    overall = pd.Series(
        True,
        index=df.index,
    )

    return pd.DataFrame(
        [
            calculate_metrics(
                df,
                before,
                "BEFORE BLACKOUT",
            ),
            calculate_metrics(
                df,
                blackout,
                "DURING BLACKOUT",
            ),
            calculate_metrics(
                df,
                after,
                "AFTER BLACKOUT",
            ),
            calculate_metrics(
                df,
                overall,
                "OVERALL",
            ),
        ]
    )


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate(
    ekf: pd.DataFrame,
):
    """
    Evaluate EKF against both raw and cleaned S-Vw4 GPS.

    Returns:
        raw_evaluated,
        clean_evaluated,
        jumps
    """

    ekf = ekf.copy()

    ekf.columns = (
        ekf.columns
        .astype(str)
        .str.strip()
    )

    ekf["timestamp_ms"] = pd.to_numeric(
        ekf["timestamp_ms"],
        errors="coerce",
    )

    ekf = ekf.dropna(
        subset=[
            "timestamp_ms",
        ]
    )

    ekf = ekf.sort_values(
        "timestamp_ms"
    ).reset_index(
        drop=True
    )

    gps = load_svw4_gps()

    clean_gps, jumps = (
        clean_gps_reference(
            gps
        )
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw_aligned = (
        align_reference_to_ekf(
            ekf,
            gps,
        )
    )

    raw = ekf.copy()

    raw["gps_x_m"] = (
        raw_aligned[
            "gps_x_m"
        ].to_numpy()
    )

    raw["gps_y_m"] = (
        raw_aligned[
            "gps_y_m"
        ].to_numpy()
    )

    raw["gps_timestamp_ms"] = (
        raw_aligned[
            "gps_timestamp_ms"
        ].to_numpy()
    )

    raw[
        "gps_timestamp_difference_ms"
    ] = raw_aligned[
        "gps_timestamp_difference_ms"
    ].to_numpy()

    raw = add_errors(
        raw
    )

    raw["gps_reference_cleaned"] = (
        False
    )

    raw["gps_jump_threshold_m"] = (
        GPS_JUMP_THRESHOLD_M
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    clean_aligned = (
        align_reference_to_ekf(
            ekf,
            clean_gps,
        )
    )

    clean = ekf.copy()

    clean["gps_x_m"] = (
        clean_aligned[
            "gps_x_m"
        ].to_numpy()
    )

    clean["gps_y_m"] = (
        clean_aligned[
            "gps_y_m"
        ].to_numpy()
    )

    clean["gps_timestamp_ms"] = (
        clean_aligned[
            "gps_timestamp_ms"
        ].to_numpy()
    )

    clean[
        "gps_timestamp_difference_ms"
    ] = clean_aligned[
        "gps_timestamp_difference_ms"
    ].to_numpy()

    clean = add_errors(
        clean
    )

    clean["gps_reference_cleaned"] = (
        True
    )

    clean["gps_jump_threshold_m"] = (
        GPS_JUMP_THRESHOLD_M
    )

    return (
        raw,
        clean,
        jumps,
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

def main():

    input_file = (
        RESULTS_DIR
        / "ekf_imu_corrected_trajectory.csv"
    )

    raw_file = (
        RESULTS_DIR
        / "ekf_position_evaluation_svw4.csv"
    )

    clean_file = (
        RESULTS_DIR
        / "ekf_position_evaluation_svw4_clean.csv"
    )

    metrics_file = (
        RESULTS_DIR
        / "ekf_metrics_svw4.txt"
    )

    print("=" * 70)
    print(
        "S-Vw4 EKF POSITION EVALUATION"
    )
    print("=" * 70)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing EKF file:\n{input_file}"
        )

    ekf = pd.read_csv(
        input_file
    )

    print()
    print(
        f"EKF rows: {len(ekf)}"
    )

    raw, clean, jumps = evaluate(
        ekf
    )

    raw.to_csv(
        raw_file,
        index=False,
    )

    clean.to_csv(
        clean_file,
        index=False,
    )

    raw_report = make_report(
        raw
    )

    clean_report = make_report(
        clean
    )

    lines = []

    lines.append(
        "S-Vw4 EKF POSITION EVALUATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")
    lines.append(
        "RAW S-Vw4 REFERENCE"
    )

    lines.append(
        raw_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    lines.append("")
    lines.append(
        "CLEANED S-Vw4 REFERENCE"
    )

    lines.append(
        clean_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    lines.append("")
    lines.append(
        "GPS JUMP DETECTION"
    )

    lines.append(
        f"Threshold: "
        f"{GPS_JUMP_THRESHOLD_M:.2f} m"
    )

    lines.append(
        f"Large GPS jumps: "
        f"{len(jumps)}"
    )

    outliers = int(
        clean_gps_outlier_count(
            ekf
        )
    )

    lines.append(
        f"Physically inconsistent "
        f"GPS points: {outliers}"
    )

    metrics_file.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "RAW S-Vw4 REFERENCE"
    )

    print(
        raw_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "CLEANED S-Vw4 REFERENCE"
    )

    print(
        clean_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"GPS jumps detected: "
        f"{len(jumps)}"
    )

    print()
    print(
        f"Raw evaluation : "
        f"{raw_file}"
    )

    print(
        f"Clean evaluation: "
        f"{clean_file}"
    )

    print(
        f"Metrics report : "
        f"{metrics_file}"
    )


def clean_gps_outlier_count(
    ekf: pd.DataFrame,
) -> int:
    """
    Return number of physically inconsistent GPS points.

    This helper reloads S-Vw4 so direct execution can report the
    count without changing the public evaluate() interface.
    """

    gps = load_svw4_gps()

    _, jumps = clean_gps_reference(
        gps
    )

    return int(
        (
            jumps[
                "gps_outlier"
            ]
        )
        .fillna(False)
        .sum()
    )


if __name__ == "__main__":
    main()