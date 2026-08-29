"""
Evaluate EKF trajectory against S-Vw4 GPS.

Compatible with evaluate_ins_svw4.py.

Produces:
    results/ekf_position_evaluation_svw4.csv
    results/ekf_position_evaluation_svw4_clean.csv
    results/ekf_metrics_svw4.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_ins_svw4 import (
    evaluate,
    make_report,
    GPS_JUMP_THRESHOLD_M,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ekf_imu_corrected_trajectory.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ekf_position_evaluation_svw4.csv"
)

CLEAN_OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ekf_position_evaluation_svw4_clean.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "ekf_metrics_svw4.txt"
)


def blackout_mask(df):
    if "gnss_available" not in df.columns:
        raise ValueError(
            "EKF output does not contain gnss_available."
        )

    values = df["gnss_available"]

    if pd.api.types.is_bool_dtype(values):
        return ~values.fillna(False)

    text = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    available = text.isin(
        ["true", "1", "yes", "y"]
    )

    return ~available


def blackout_diagnostic(df):
    mask = blackout_mask(df)

    b = df.loc[mask].copy()

    if b.empty:
        return None

    t = (
        b["timestamp_ms"]
        .to_numpy(dtype=float)
    )

    x = (
        b["x_m"]
        .to_numpy(dtype=float)
    )

    y = (
        b["y_m"]
        .to_numpy(dtype=float)
    )

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])

    distance = float(
        np.hypot(dx, dy)
    )

    result = {
        "rows": len(b),
        "duration_s": float(
            (t[-1] - t[0]) / 1000.0
        ),
        "dx_m": dx,
        "dy_m": dy,
        "distance_m": distance,
    }

    if "ekf_speed_mps" in b.columns:
        speed = pd.to_numeric(
            b["ekf_speed_mps"],
            errors="coerce",
        )

        result["speed_mean_mps"] = float(
            speed.mean()
        )

        result["speed_first_mps"] = float(
            speed.iloc[0]
        )

        result["speed_last_mps"] = float(
            speed.iloc[-1]
        )

    if (
        "vx_mps" in b.columns
        and "vy_mps" in b.columns
    ):
        vx = pd.to_numeric(
            b["vx_mps"],
            errors="coerce",
        ).to_numpy(dtype=float)

        vy = pd.to_numeric(
            b["vy_mps"],
            errors="coerce",
        ).to_numpy(dtype=float)

        result["direction_first_deg"] = float(
            np.degrees(
                np.arctan2(
                    vx[0],
                    vy[0],
                )
            )
            % 360.0
        )

        result["direction_last_deg"] = float(
            np.degrees(
                np.arctan2(
                    vx[-1],
                    vy[-1],
                )
            )
            % 360.0
        )

    return result


def main():
    print("=" * 70)
    print("S-Vw4 EKF POSITION EVALUATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"EKF trajectory not found:\n{INPUT_FILE}"
        )

    ekf = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"EKF rows: {len(ekf)}"
    )

    # --------------------------------------------------------
    # evaluate_ins_svw4.py returns:
    #
    #   raw_evaluated
    #   clean_evaluated
    #   jumps
    # --------------------------------------------------------

    raw_evaluated, clean_evaluated, jumps = evaluate(
        ekf
    )

    # --------------------------------------------------------
    # Reports.
    # --------------------------------------------------------

    raw_report = make_report(
        raw_evaluated
    )

    clean_report = make_report(
        clean_evaluated
    )

    # --------------------------------------------------------
    # Save CSVs.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_evaluated.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    clean_evaluated.to_csv(
        CLEAN_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Blackout diagnostics.
    # --------------------------------------------------------

    raw_diag = blackout_diagnostic(
        raw_evaluated
    )

    clean_diag = blackout_diagnostic(
        clean_evaluated
    )

    # --------------------------------------------------------
    # Metrics report.
    # --------------------------------------------------------

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
        f"Detected jumps: {len(jumps)}"
    )

    lines.append("")
    lines.append(
        "BLACKOUT DIAGNOSTIC"
    )

    if clean_diag is not None:

        lines.append(
            f"rows={clean_diag['rows']}"
        )

        lines.append(
            f"duration_s="
            f"{clean_diag['duration_s']:.3f}"
        )

        lines.append(
            f"EKF dx="
            f"{clean_diag['dx_m']:.6f}"
        )

        lines.append(
            f"EKF dy="
            f"{clean_diag['dy_m']:.6f}"
        )

        lines.append(
            f"EKF distance="
            f"{clean_diag['distance_m']:.6f}"
        )

        if "speed_mean_mps" in clean_diag:

            lines.append(
                f"speed mean="
                f"{clean_diag['speed_mean_mps']:.6f}"
            )

            lines.append(
                f"speed first="
                f"{clean_diag['speed_first_mps']:.6f}"
            )

            lines.append(
                f"speed last="
                f"{clean_diag['speed_last_mps']:.6f}"
            )

        if "direction_first_deg" in clean_diag:

            lines.append(
                f"direction first="
                f"{clean_diag['direction_first_deg']:.6f}"
            )

            lines.append(
                f"direction last="
                f"{clean_diag['direction_last_deg']:.6f}"
            )

    else:

        lines.append(
            "No GNSS blackout detected."
        )

    lines.append("")
    lines.append(
        "GPS TIMESTAMP ALIGNMENT"
    )

    max_difference = pd.to_numeric(
        raw_evaluated[
            "gps_timestamp_difference_ms"
        ],
        errors="coerce",
    ).max()

    lines.append(
        "Maximum GPS timestamp difference: "
        f"{max_difference} ms"
    )

    # --------------------------------------------------------
    # EKF Q/R.
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "EKF PARAMETERS"
    )

    if "process_noise_q" in raw_evaluated.columns:

        q = (
            raw_evaluated[
                "process_noise_q"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        lines.append(
            f"Q={q}"
        )

    if "measurement_noise_r" in raw_evaluated.columns:

        r = (
            raw_evaluated[
                "measurement_noise_r"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        lines.append(
            f"R={r}"
        )

    METRICS_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console.
    # --------------------------------------------------------

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
        f"GPS jumps detected: {len(jumps)}"
    )

    print()
    print(
        "BLACKOUT DIAGNOSTIC"
    )

    if clean_diag is not None:

        print(
            f"rows              : "
            f"{clean_diag['rows']}"
        )

        print(
            f"duration           : "
            f"{clean_diag['duration_s']:.3f} s"
        )

        print(
            f"EKF dx             : "
            f"{clean_diag['dx_m']:.3f} m"
        )

        print(
            f"EKF dy             : "
            f"{clean_diag['dy_m']:.3f} m"
        )

        print(
            f"EKF displacement   : "
            f"{clean_diag['distance_m']:.3f} m"
        )

        if "speed_mean_mps" in clean_diag:

            print(
                f"speed mean         : "
                f"{clean_diag['speed_mean_mps']:.3f} m/s"
            )

            print(
                f"speed first        : "
                f"{clean_diag['speed_first_mps']:.3f} m/s"
            )

            print(
                f"speed last         : "
                f"{clean_diag['speed_last_mps']:.3f} m/s"
            )

    print()
    print(
        "Saved:"
    )

    print(
        f"Raw   : {OUTPUT_FILE}"
    )

    print(
        f"Clean : {CLEAN_OUTPUT_FILE}"
    )

    print(
        f"Report: {METRICS_FILE}"
    )


if __name__ == "__main__":
    main()