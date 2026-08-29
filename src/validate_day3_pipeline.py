"""Final structural validation for reproducible AI, INS, and EKF artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import MEASUREMENT_NOISE_R, PROCESS_NOISE_Q
from evaluate_ins_svw4 import source_gps


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS = PROJECT_ROOT / "results"

SOURCE_SVW4 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_NAVIGATION_ROWS = 126477
EXPECTED_AI_ROWS = 12648
EXPECTED_BLACKOUT_ROWS = 200

AI_TIMESTAMP_TOLERANCE_MS = 60


# ============================================================
# GENERAL FRAME CHECK
# ============================================================

def check_frame(
    name: str,
    frame: pd.DataFrame,
    required: list[str],
    report: list[str],
    errors: list[str],
    sparse_numeric_columns: set[str] | None = None,
) -> None:
    """
    Validate required structure and numeric data.

    Some EKF columns are intentionally sparse because AI
    predictions exist only at the original 12,648 AI timestamps.

    Those sparse columns are checked separately against the
    actual AI timestamps.
    """

    if sparse_numeric_columns is None:
        sparse_numeric_columns = set()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        errors.append(
            f"{name}: missing columns {missing}"
        )
        return

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_columns = (
        frame.select_dtypes(
            include=[np.number]
        ).columns
    )

    for column in numeric_columns:

        values = frame[column].to_numpy(
            dtype=float
        )

        # Sparse columns are checked separately.
        if column in sparse_numeric_columns:

            if np.isinf(values).any():

                errors.append(
                    f"{name}: Inf detected in sparse "
                    f"column {column}"
                )

            continue

        # Normal numeric columns must be finite.
        if not np.isfinite(values).all():

            errors.append(
                f"{name}: NaN/Inf detected in {column}"
            )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if frame["timestamp_ms"].duplicated().any():

        errors.append(
            f"{name}: duplicate timestamps"
        )

    if not frame[
        "timestamp_ms"
    ].is_monotonic_increasing:

        errors.append(
            f"{name}: timestamps not ordered"
        )

    # --------------------------------------------------------
    # Timing statistics
    # --------------------------------------------------------

    dt = (
        frame[
            "timestamp_ms"
        ]
        .diff()
        .dropna()
        / 1000.0
    )

    if not dt.empty:

        if (dt < 0).any():

            errors.append(
                f"{name}: negative dt"
            )

        report.append(
            f"{name}: rows={len(frame)}, "
            f"dt_s min/median/max="
            f"{dt.min():.3f}/"
            f"{dt.median():.3f}/"
            f"{dt.max():.3f}"
        )

    else:

        report.append(
            f"{name}: rows={len(frame)}, "
            "single sample"
        )


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source_svw4() -> pd.DataFrame:
    """Load repository S-Vw4 source."""

    if not SOURCE_SVW4.exists():

        raise FileNotFoundError(
            f"S-Vw4 source file not found:\n"
            f"{SOURCE_SVW4}"
        )

    source = pd.read_csv(
        SOURCE_SVW4,
        encoding="latin1",
    )

    source.columns = (
        source.columns
        .astype(str)
        .str.strip()
    )

    return source


# ============================================================
# VALIDATE SPARSE AI COLUMNS
# ============================================================

def validate_sparse_ai_columns(
    ekf: pd.DataFrame,
    ai: pd.DataFrame,
    report: list[str],
    errors: list[str],
) -> None:
    """
    Validate sparse AI-derived EKF columns.

    AI predictions exist at only 12,648 timestamps.

    Therefore these columns are expected to be:

        finite  -> at actual AI timestamps
        NaN     -> at non-AI navigation timestamps

    Columns checked:
        ai_speed_mps
        speed_confidence
        selected_speed_mps
        speed_error_mps
    """

    sparse_columns = [
        "ai_speed_mps",
        "speed_confidence",
        "selected_speed_mps",
        "speed_error_mps",
    ]

    for column in sparse_columns:

        if column not in ekf.columns:

            errors.append(
                f"EKF: missing sparse AI column "
                f"{column}"
            )

    if errors:
        return

    # --------------------------------------------------------
    # Actual AI timestamps
    # --------------------------------------------------------

    ai_timestamps = (
        ai[
            "timestamp_ms"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    ekf_timestamps = (
        ekf[
            "timestamp_ms"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    # --------------------------------------------------------
    # Find exact/near AI timestamp rows
    # --------------------------------------------------------

    ekf_ai_mask = np.zeros(
        len(ekf),
        dtype=bool,
    )

    matched_indices = []

    for timestamp in ai_timestamps:

        positions = np.searchsorted(
            ekf_timestamps,
            timestamp,
        )

        candidates = []

        if positions < len(
            ekf_timestamps
        ):
            candidates.append(
                positions
            )

        if positions > 0:
            candidates.append(
                positions - 1
            )

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda index: abs(
                int(
                    ekf_timestamps[index]
                )
                - int(timestamp)
            ),
        )

        difference = abs(
            int(
                ekf_timestamps[best]
            )
            - int(timestamp)
        )

        if (
            difference
            <= AI_TIMESTAMP_TOLERANCE_MS
        ):

            ekf_ai_mask[best] = True
            matched_indices.append(
                best
            )

    matched_count = int(
        ekf_ai_mask.sum()
    )

    report.append(
        "EKF AI timestamp matches: "
        f"{matched_count}/{len(ai)}"
    )

    if matched_count != len(ai):

        errors.append(
            "EKF does not contain all "
            "12,648 actual AI timestamps."
        )

        return

    # --------------------------------------------------------
    # Validate sparse columns
    # --------------------------------------------------------

    for column in sparse_columns:

        values = ekf[
            column
        ].to_numpy(
            dtype=float
        )

        ai_values = values[
            ekf_ai_mask
        ]

        non_ai_values = values[
            ~ekf_ai_mask
        ]

        # Actual AI rows must be finite.
        if not np.isfinite(
            ai_values
        ).all():

            errors.append(
                f"EKF: {column} contains "
                "NaN/Inf at an actual "
                "AI timestamp."
            )

        # Non-AI rows may be NaN, but never Inf.
        if np.isinf(
            non_ai_values
        ).any():

            errors.append(
                f"EKF: {column} contains "
                "Inf on non-AI rows."
            )

    # --------------------------------------------------------
    # Expected sparse counts
    # --------------------------------------------------------

    for column in [
        "ai_speed_mps",
        "speed_confidence",
        "selected_speed_mps",
        "speed_error_mps",
    ]:

        finite_count = int(
            np.isfinite(
                ekf[
                    column
                ].to_numpy(
                    dtype=float
                )
            ).sum()
        )

        report.append(
            f"EKF {column}: "
            f"finite AI rows="
            f"{finite_count}/{EXPECTED_AI_ROWS}"
        )

        if finite_count != (
            EXPECTED_AI_ROWS
        ):

            errors.append(
                f"EKF {column}: expected "
                f"{EXPECTED_AI_ROWS} finite "
                f"AI rows, found "
                f"{finite_count}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    errors: list[str] = []

    report: list[str] = []

    # ========================================================
    # LOAD PIPELINE ARTIFACTS
    # ========================================================

    ai_file = (
        RESULTS
        / "ai_speed_output.csv"
    )

    navigation_file = (
        RESULTS
        / "navigation_gnss_blackout.csv"
    )

    ins_file = (
        RESULTS
        / "ins_trajectory.csv"
    )

    ekf_file = (
        RESULTS
        / "ekf_imu_corrected_trajectory.csv"
    )

    required_files = {
        "AI": ai_file,
        "navigation": navigation_file,
        "INS": ins_file,
        "EKF": ekf_file,
    }

    frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for name, path in (
        required_files.items()
    ):

        if not path.exists():

            errors.append(
                f"{name}: missing required "
                f"file {path}"
            )

        else:

            frames[name] = pd.read_csv(
                path
            )

    # --------------------------------------------------------
    # Missing file failure
    # --------------------------------------------------------

    if errors:

        text = (
            "\n".join(
                report
                + ["ERRORS:"]
                + errors
            )
            + "\n"
        )

        output_file = (
            RESULTS
            / "day3_final_validation.txt"
        )

        output_file.write_text(
            text,
            encoding="utf-8",
        )

        print(
            text,
            end="",
        )

        print(
            f"Saved validation report: "
            f"{output_file}"
        )

        raise SystemExit(1)

    ai = frames["AI"]

    navigation = frames[
        "navigation"
    ]

    ins = frames["INS"]

    ekf = frames["EKF"]

    # ========================================================
    # BASIC FRAME VALIDATION
    # ========================================================

    check_frame(
        "AI",
        ai,
        [
            "timestamp_ms",
            "ai_speed_mps",
            "speed_confidence",
        ],
        report,
        errors,
    )

    # Navigation itself may contain sparse AI values.
    check_frame(
        "navigation",
        navigation,
        [
            "timestamp_ms",
            "ai_speed_mps",
            "heading_deg",
            "gnss_available",
        ],
        report,
        errors,
        sparse_numeric_columns={
            "ai_speed_mps",
        },
    )

    check_frame(
        "INS",
        ins,
        [
            "timestamp_ms",
            "dt_s",
            "x_m",
            "y_m",
        ],
        report,
        errors,
    )

    # EKF contains intentionally sparse AI fields.
    check_frame(
        "EKF",
        ekf,
        [
            "timestamp_ms",
            "dt_s",
            "x_m",
            "y_m",
            "process_noise_q",
            "measurement_noise_r",
        ],
        report,
        errors,
        sparse_numeric_columns={
            "ai_speed_mps",
            "speed_confidence",
            "selected_speed_mps",
            "speed_error_mps",
            "corrected_speed_mps",
        },
    )

    # ========================================================
    # ROW COUNTS
    # ========================================================

    if len(navigation) != (
        EXPECTED_NAVIGATION_ROWS
    ):

        errors.append(
            f"Navigation rows: expected "
            f"{EXPECTED_NAVIGATION_ROWS}, "
            f"found {len(navigation)}"
        )

    if len(ai) != (
        EXPECTED_AI_ROWS
    ):

        errors.append(
            f"AI rows: expected "
            f"{EXPECTED_AI_ROWS}, "
            f"found {len(ai)}"
        )

    if len(ekf) != (
        EXPECTED_NAVIGATION_ROWS
    ):

        errors.append(
            f"EKF rows: expected "
            f"{EXPECTED_NAVIGATION_ROWS}, "
            f"found {len(ekf)}"
        )

    # ========================================================
    # AI OUTPUT TIMESTAMP VALIDATION
    # ========================================================

    try:

        expected = (
            source_gps()[
                "gps_timestamp_ms"
            ]
            .iloc[49::10]
            .to_numpy(
                dtype=np.int64
            )
        )

        actual = (
            ai[
                "timestamp_ms"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if not np.array_equal(
            actual,
            expected,
        ):

            errors.append(
                "AI timestamps do not follow "
                "source last-window-sample "
                "convention"
            )

        report.append(
            "AI timestamps: "
            f"predictions={len(ai)}, "
            f"first="
            f"{ai.timestamp_ms.iloc[0]}, "
            f"last="
            f"{ai.timestamp_ms.iloc[-1]}, "
            f"unique="
            f"{not ai.timestamp_ms.duplicated().any()}"
        )

    except Exception as exc:

        errors.append(
            "AI timestamp validation "
            f"failed: {exc}"
        )

    # ========================================================
    # S-Vw4 SPEED ALIGNMENT
    # ========================================================

    try:

        source_speed = (
            load_source_svw4()
        )

        required_speed_columns = [
            "TIME SINCE START (ms)",
            "GPS SPEED (Kmh)",
        ]

        missing_speed_columns = [
            column
            for column in (
                required_speed_columns
            )
            if column
            not in source_speed.columns
        ]

        if missing_speed_columns:

            errors.append(
                "S-Vw4 missing speed "
                "columns: "
                f"{missing_speed_columns}"
            )

        else:

            gps_speed = (
                source_speed[
                    [
                        "TIME SINCE START (ms)",
                        "GPS SPEED (Kmh)",
                    ]
                ]
                .rename(
                    columns={
                        "TIME SINCE START (ms)":
                            "timestamp_ms"
                    }
                )
                .copy()
            )

            gps_speed[
                "timestamp_ms"
            ] = pd.to_numeric(
                gps_speed[
                    "timestamp_ms"
                ],
                errors="coerce",
            )

            gps_speed[
                "GPS SPEED (Kmh)"
            ] = pd.to_numeric(
                gps_speed[
                    "GPS SPEED (Kmh)"
                ],
                errors="coerce",
            )

            gps_speed = (
                gps_speed
                .dropna(
                    subset=[
                        "timestamp_ms",
                        "GPS SPEED (Kmh)",
                    ]
                )
                .sort_values(
                    "timestamp_ms"
                )
            )

            speed_join = (
                pd.merge_asof(
                    ai.sort_values(
                        "timestamp_ms"
                    ),
                    gps_speed,
                    on="timestamp_ms",
                    direction="nearest",
                    tolerance=60,
                )
                .dropna(
                    subset=[
                        "GPS SPEED (Kmh)"
                    ]
                )
            )

            speed_error = (
                speed_join[
                    "ai_speed_mps"
                ]
                -
                speed_join[
                    "GPS SPEED (Kmh)"
                ] / 3.6
            )

            report.append(
                "AI/S-Vw4 speed units: "
                f"matched="
                f"{len(speed_join)}/"
                f"{len(ai)}, "
                f"MAE_mps="
                f"{abs(speed_error).mean():.4f}, "
                f"RMSE_mps="
                f"{np.sqrt(np.mean(speed_error ** 2)):.4f}"
            )

            if len(speed_join) != (
                len(ai)
            ):

                errors.append(
                    "Some AI predictions "
                    "did not align with "
                    "S-Vw4 speed timestamps."
                )

    except Exception as exc:

        errors.append(
            "AI/S-Vw4 speed validation "
            f"failed: {exc}"
        )

    # ========================================================
    # GNSS BLACKOUT VALIDATION
    # ========================================================

    try:

        gnss_values = (
            navigation[
                "gnss_available"
            ]
        )

        # Correct handling if column was loaded as strings.
        if (
            gnss_values.dtype
            == object
        ):

            gnss_values = (
                gnss_values
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                )
                .fillna(False)
                .astype(bool)
            )

        else:

            gnss_values = (
                gnss_values
                .fillna(False)
                .astype(bool)
            )

        blackout = (
            ~gnss_values
        )

        blackout_count = int(
            blackout.sum()
        )

        if blackout_count > 0:

            blackout_start = int(
                navigation.loc[
                    blackout,
                    "timestamp_ms",
                ].iloc[0]
            )

            blackout_end = int(
                navigation.loc[
                    blackout,
                    "timestamp_ms",
                ].iloc[-1]
            )

            blackout_duration = (
                blackout_end
                - blackout_start
            ) / 1000.0

            report.append(
                "blackout: "
                f"samples="
                f"{blackout_count}, "
                f"first-to-last_s="
                f"{blackout_duration:.3f}"
            )

        else:

            report.append(
                "blackout: samples=0"
            )

        if blackout_count != (
            EXPECTED_BLACKOUT_ROWS
        ):

            errors.append(
                f"Blackout is "
                f"{blackout_count} samples; "
                "expected exactly 200."
            )

    except Exception as exc:

        errors.append(
            "GNSS blackout validation "
            f"failed: {exc}"
        )

    # ========================================================
    # EKF Q/R VALIDATION
    # ========================================================

    try:

        q_values = (
            ekf[
                "process_noise_q"
            ]
            .dropna()
            .unique()
        )

        r_values = (
            ekf[
                "measurement_noise_r"
            ]
            .dropna()
            .unique()
        )

        report.append(
            f"EKF Q="
            f"{q_values.tolist()}, "
            f"R="
            f"{r_values.tolist()}"
        )

        if (
            len(q_values) != 1
            or not np.isclose(
                q_values[0],
                PROCESS_NOISE_Q,
            )
        ):

            errors.append(
                "EKF Q differs from "
                f"required value "
                f"{PROCESS_NOISE_Q}"
            )

        if (
            len(r_values) != 1
            or not np.isclose(
                r_values[0],
                MEASUREMENT_NOISE_R,
            )
        ):

            errors.append(
                "EKF R differs from "
                f"required value "
                f"{MEASUREMENT_NOISE_R}"
            )

    except Exception as exc:

        errors.append(
            "EKF Q/R validation "
            f"failed: {exc}"
        )

    # ========================================================
    # EKF BLACKOUT VALIDATION
    # ========================================================

    if (
        "gnss_available"
        in ekf.columns
    ):

        ekf_gnss = (
            ekf[
                "gnss_available"
            ]
        )

        if (
            ekf_gnss.dtype
            == object
        ):

            ekf_gnss = (
                ekf_gnss
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                )
                .fillna(False)
                .astype(bool)
            )

        else:

            ekf_gnss = (
                ekf_gnss
                .fillna(False)
                .astype(bool)
            )

        ekf_blackout = (
            ~ekf_gnss
        )

        ekf_blackout_count = int(
            ekf_blackout.sum()
        )

        report.append(
            "EKF blackout samples="
            f"{ekf_blackout_count}"
        )

        if ekf_blackout_count != (
            EXPECTED_BLACKOUT_ROWS
        ):

            errors.append(
                "EKF blackout sample "
                f"count is "
                f"{ekf_blackout_count}; "
                "expected 200."
            )

    else:

        errors.append(
            "EKF missing "
            "gnss_available column."
        )

    # ========================================================
    # EKF STATE FINITE VALIDATION
    # ========================================================

    ekf_state_columns = [
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "ekf_speed_mps",
        "dt_s",
    ]

    for column in (
        ekf_state_columns
    ):

        if column not in ekf.columns:

            errors.append(
                f"EKF missing state "
                f"column {column}"
            )

            continue

        values = ekf[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            errors.append(
                f"EKF state column "
                f"{column} contains "
                "NaN/Inf"
            )

    # ========================================================
    # SPARSE AI COLUMN VALIDATION
    # ========================================================

    validate_sparse_ai_columns(
        ekf,
        ai,
        report,
        errors,
    )

    # ========================================================
    # CORRECTED SPEED VALIDATION
    # ========================================================

    if (
        "corrected_speed_mps"
        in ekf.columns
    ):

        corrected = ekf[
            "corrected_speed_mps"
        ].to_numpy(
            dtype=float
        )

        if np.isinf(
            corrected
        ).any():

            errors.append(
                "EKF corrected_speed_mps "
                "contains Inf"
            )

        fallback_rows = (
            ekf[
                "speed_source"
            ].eq(
                "AI_FALLBACK"
            )
            if "speed_source"
            in ekf.columns
            else pd.Series(
                False,
                index=ekf.index,
            )
        )

        expected_fallback = int(
            fallback_rows.sum()
        )

        actual_nan = int(
            np.isnan(
                corrected
            ).sum()
        )

        report.append(
            "EKF corrected speed: "
            f"NaN rows="
            f"{actual_nan}, "
            f"AI fallback rows="
            f"{expected_fallback}"
        )

        # The current runner intentionally leaves
        # corrected_speed_mps empty and uses AI fallback.
        #
        # Therefore NaN is valid here.

    # ========================================================
    # AI MANIFEST VALIDATION
    # ========================================================

    manifest = (
        RESULTS
        / "ai_speed_output_manifest.txt"
    )

    if not manifest.exists():

        errors.append(
            "AI output reproducibility "
            "manifest is missing."
        )

    else:

        manifest_text = (
            manifest.read_text(
                encoding="utf-8"
            )
        )

        source_string = (
            "source_file="
            + str(SOURCE_SVW4)
        )

        if source_string not in (
            manifest_text
        ):

            errors.append(
                "AI output manifest does "
                "not identify the repository "
                "S-Vw4 source."
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    if errors:

        report.append(
            "ERRORS:"
        )

        report.extend(
            errors
        )

    else:

        report.append(
            "VALIDATION: PASS"
        )

    text = (
        "\n".join(
            report
        )
        + "\n"
    )

    output_file = (
        RESULTS
        / "day3_final_validation.txt"
    )

    output_file.write_text(
        text,
        encoding="utf-8",
    )

    print(
        text,
        end="",
    )

    print(
        f"Saved validation report: "
        f"{output_file}"
    )

    if errors:

        raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()