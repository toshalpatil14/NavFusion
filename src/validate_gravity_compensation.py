"""Validate existing phone-frame gravity compensation. Does not rewrite IMU CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_FILE = PROJECT_ROOT / "results" / "imu_preprocessing.csv"
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv"
REPORT_FILE = PROJECT_ROOT / "results" / "gravity_compensation_validation.txt"

# Same raw names as src/inspect_imu_preprocessing.py
TIMESTAMP_COL = "TIME SINCE START (ms)"
ACC_X_COL = "ACCELEROMETER X (m/s²)"
ACC_Y_COL = "ACCELEROMETER Y (m/s²)"
ACC_Z_COL = "ACCELEROMETER Z (m/s²)"
GRAV_X_COL = "GRAVITY X (m/s²)"
GRAV_Y_COL = "GRAVITY Y (m/s²)"
GRAV_Z_COL = "GRAVITY Z (m/s²)"

STANDARD_G = 9.80665  # conventional standard gravity, m/s^2
# Earth surface gravity is about 9.78–9.83. Larger mean error is sensor/scale, not geography.
EARTH_G_BAND = 0.05
# Consumer MEMS gravity-sensor magnitude is typically within a few percent of g.
SENSOR_G_BAND = 0.5
# Reconstruction must match the stored linear_* columns (CSV float noise only).
RECONSTRUCTION_ATOL = 1e-6
REQUIRED_PREPROCESSED = [
    "timestamp_ms",
    "linear_ax",
    "linear_ay",
    "linear_az",
    "linear_accel_magnitude",
    "pitch_deg",
    "roll_deg",
]


def stats_block(name: str, values: np.ndarray) -> list[str]:
    return [
        f"{name}",
        f"  min   : {np.min(values):.6f}",
        f"  mean  : {np.mean(values):.6f}",
        f"  median: {np.median(values):.6f}",
        f"  std   : {np.std(values):.6f}",
        f"  max   : {np.max(values):.6f}",
    ]


def angular_abs_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def gravity_implied_pitch_roll_deg(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pitch/roll from the gravity vector in Android device axes.

    Gravity is reported in the same phone frame as the accelerometer.
    When the device is screen-up and at rest, gravity is approximately (0, 0, +g).
    Pitch is rotation about Y: atan2(-gx, hypot(gy, gz)).
    Roll is rotation about X: atan2(gy, gz).
    """
    pitch = np.degrees(np.arctan2(-gx, np.hypot(gy, gz)))
    roll = np.degrees(np.arctan2(gy, gz))
    return pitch, roll


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    report: list[str] = []

    report.append("=" * 60)
    report.append("GRAVITY COMPENSATION VALIDATION")
    report.append("=" * 60)
    report.append("Check: linear_acceleration = raw_acceleration - gravity")
    report.append("Preprocessed file is not modified.")
    report.append("")

    if not PREPROCESSED_FILE.exists():
        errors.append(f"Missing preprocessed IMU file: {PREPROCESSED_FILE}")
        _write_and_print(report, errors, warnings)
        raise SystemExit(1)

    imu = pd.read_csv(PREPROCESSED_FILE)
    missing = [c for c in REQUIRED_PREPROCESSED if c not in imu.columns]
    if missing:
        errors.append(f"imu_preprocessing.csv missing columns: {missing}")
        _write_and_print(report, errors, warnings)
        raise SystemExit(1)

    n_pre = len(imu)
    report.append(f"Preprocessed rows : {n_pre}")
    report.append(f"Preprocessed file : {PREPROCESSED_FILE}")

    numeric_imu = imu[REQUIRED_PREPROCESSED].apply(pd.to_numeric, errors="coerce")
    imu_ok = np.isfinite(numeric_imu.to_numpy(dtype=float)).all()
    report.append(f"Preprocessed NaN/Inf: {'PASSED' if imu_ok else 'FAILED'}")
    if not imu_ok:
        errors.append("imu_preprocessing.csv contains NaN or Inf in required columns.")

    if not RAW_FILE.exists():
        errors.append(f"Missing raw S-Vw4 file: {RAW_FILE}")
        _write_and_print(report, errors, warnings)
        raise SystemExit(1)

    raw = pd.read_csv(RAW_FILE, encoding="latin1")
    raw.columns = raw.columns.str.strip()
    raw_required = [
        TIMESTAMP_COL,
        ACC_X_COL,
        ACC_Y_COL,
        ACC_Z_COL,
        GRAV_X_COL,
        GRAV_Y_COL,
        GRAV_Z_COL,
    ]
    missing_raw = [c for c in raw_required if c not in raw.columns]
    if missing_raw:
        errors.append(f"S-Vw4.csv missing columns: {missing_raw}")
        _write_and_print(report, errors, warnings)
        raise SystemExit(1)

    raw_num = raw[raw_required].apply(pd.to_numeric, errors="coerce")
    raw_ok = np.isfinite(raw_num.to_numpy(dtype=float)).all()
    report.append(f"Raw sensor NaN/Inf : {'PASSED' if raw_ok else 'FAILED'}")
    if not raw_ok:
        errors.append("Raw accelerometer/gravity columns contain NaN or Inf.")

    raw_num = raw_num.rename(columns={TIMESTAMP_COL: "timestamp_ms"})
    imu_join = imu.copy()
    imu_join["timestamp_ms"] = pd.to_numeric(imu_join["timestamp_ms"], errors="coerce")
    merged = pd.merge(imu_join, raw_num, on="timestamp_ms", how="inner", validate="one_to_one")
    report.append(f"Raw rows           : {len(raw)}")
    report.append(f"Timestamp-aligned  : {len(merged)}")

    if len(merged) != n_pre:
        errors.append(
            f"Timestamp alignment dropped rows: preprocessed={n_pre}, aligned={len(merged)}."
        )
    if merged.empty:
        errors.append("No timestamps matched between imu_preprocessing.csv and S-Vw4.csv.")
        _write_and_print(report, errors, warnings)
        raise SystemExit(1)

    acc_x = merged[ACC_X_COL].to_numpy(dtype=float)
    acc_y = merged[ACC_Y_COL].to_numpy(dtype=float)
    acc_z = merged[ACC_Z_COL].to_numpy(dtype=float)
    grav_x = merged[GRAV_X_COL].to_numpy(dtype=float)
    grav_y = merged[GRAV_Y_COL].to_numpy(dtype=float)
    grav_z = merged[GRAV_Z_COL].to_numpy(dtype=float)
    lin_x = merged["linear_ax"].to_numpy(dtype=float)
    lin_y = merged["linear_ay"].to_numpy(dtype=float)
    lin_z = merged["linear_az"].to_numpy(dtype=float)
    lin_mag_stored = merged["linear_accel_magnitude"].to_numpy(dtype=float)
    pitch = merged["pitch_deg"].to_numpy(dtype=float)
    roll = merged["roll_deg"].to_numpy(dtype=float)

    recon_x = acc_x - grav_x
    recon_y = acc_y - grav_y
    recon_z = acc_z - grav_z
    recon_err = np.column_stack(
        [np.abs(lin_x - recon_x), np.abs(lin_y - recon_y), np.abs(lin_z - recon_z)]
    )
    recon_max = float(np.max(recon_err))
    recon_mean = float(np.mean(recon_err))
    reconstruction_ok = recon_max <= RECONSTRUCTION_ATOL
    report.append("")
    report.append("-" * 60)
    report.append("RECONSTRUCTION: linear = accelerometer - gravity")
    report.append("-" * 60)
    report.append(f"Max |stored - reconstructed| : {recon_max:.3e} m/s^2")
    report.append(f"Mean |stored - reconstructed|: {recon_mean:.3e} m/s^2")
    report.append(f"Tolerance                    : {RECONSTRUCTION_ATOL:.0e} m/s^2")
    report.append(f"Reconstruction               : {'PASSED' if reconstruction_ok else 'FAILED'}")
    if not reconstruction_ok:
        errors.append(
            "Stored linear acceleration does not equal accelerometer minus gravity."
        )

    lin_mag = np.sqrt(lin_x**2 + lin_y**2 + lin_z**2)
    mag_err = np.abs(lin_mag_stored - lin_mag)
    mag_ok = float(np.max(mag_err)) <= RECONSTRUCTION_ATOL
    report.append(f"linear_accel_magnitude check : {'PASSED' if mag_ok else 'FAILED'}")
    if not mag_ok:
        errors.append("Stored linear_accel_magnitude does not match sqrt(ax²+ay²+az²).")

    grav_mag = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2)
    grav_mean = float(np.mean(grav_mag))
    grav_std = float(np.std(grav_mag))
    grav_dev = grav_mean - STANDARD_G
    abs_dev = abs(grav_dev)

    report.append("")
    report.append("-" * 60)
    report.append("RAW GRAVITY")
    report.append("-" * 60)
    report.extend(stats_block("Gravity X (m/s^2)", grav_x))
    report.extend(stats_block("Gravity Y (m/s^2)", grav_y))
    report.extend(stats_block("Gravity Z (m/s^2)", grav_z))
    report.extend(stats_block("Gravity magnitude (m/s^2)", grav_mag))
    report.append("")
    report.append(f"Standard gravity g0          : {STANDARD_G:.5f} m/s^2")
    report.append(f"Mean gravity magnitude       : {grav_mean:.6f} m/s^2")
    report.append(f"Deviation from g0            : {grav_dev:+.6f} m/s^2")
    report.append(f"Gravity magnitude std        : {grav_std:.6f} m/s^2")
    report.append(
        "Note: gravity magnitude should stay near Earth g. "
        "It is independent of vehicle acceleration."
    )

    if abs_dev > SENSOR_G_BAND:
        errors.append(
            f"Mean gravity magnitude {grav_mean:.4f} m/s^2 is more than "
            f"{SENSOR_G_BAND} m/s^2 from g0={STANDARD_G}. Not consistent with terrestrial gravity."
        )
        gravity_status = "FAILED"
    elif abs_dev > EARTH_G_BAND:
        warnings.append(
            f"Mean gravity magnitude differs from g0 by {abs_dev:.4f} m/s^2 "
            f"(beyond Earth variation ~{EARTH_G_BAND} m/s^2, within typical MEMS {SENSOR_G_BAND} m/s^2)."
        )
        gravity_status = "WARNING"
    else:
        gravity_status = "PASSED"
    report.append(f"Gravity magnitude vs g0      : {gravity_status}")

    # Magnitude of the gravity vector should be stable even as the phone attitude changes.
    if grav_std > SENSOR_G_BAND:
        errors.append(
            f"Gravity magnitude std {grav_std:.4f} m/s^2 exceeds {SENSOR_G_BAND} m/s^2."
        )
    elif grav_std > EARTH_G_BAND:
        warnings.append(
            f"Gravity magnitude std {grav_std:.4f} m/s^2 is larger than Earth-g variation "
            f"({EARTH_G_BAND} m/s^2); the gravity sensor may be noisy."
        )

    report.append("")
    report.append("-" * 60)
    report.append("LINEAR ACCELERATION (MOTION RESIDUAL AFTER GRAVITY REMOVAL)")
    report.append("-" * 60)
    report.append(
        "These values are vehicle/phone motion plus sensor noise, not a gravity error."
    )
    report.extend(stats_block("linear_ax (m/s^2)", lin_x))
    report.extend(stats_block("linear_ay (m/s^2)", lin_y))
    report.extend(stats_block("linear_az (m/s^2)", lin_z))
    report.extend(stats_block("linear acceleration magnitude (m/s^2)", lin_mag))
    report.append("")
    report.append(f"Residual mean (magnitude)    : {float(np.mean(lin_mag)):.6f} m/s^2")
    report.append(f"Residual median (magnitude)  : {float(np.median(lin_mag)):.6f} m/s^2")
    report.append(f"Residual std (magnitude)     : {float(np.std(lin_mag)):.6f} m/s^2")
    report.append(
        f"Per-axis residual mean       : "
        f"x={float(np.mean(lin_x)):.6f}, y={float(np.mean(lin_y)):.6f}, z={float(np.mean(lin_z)):.6f}"
    )

    implied_pitch, implied_roll = gravity_implied_pitch_roll_deg(grav_x, grav_y, grav_z)
    pitch_err = angular_abs_error_deg(pitch, implied_pitch)
    roll_err = angular_abs_error_deg(roll, implied_roll)
    tilt_from_z = np.degrees(np.arctan2(np.hypot(grav_x, grav_y), grav_z))

    report.append("")
    report.append("-" * 60)
    report.append("PITCH/ROLL vs GRAVITY DIRECTION")
    report.append("-" * 60)
    report.append(
        "Gravity compensation uses GRAVITY X/Y/Z, not pitch/roll. "
        "This section only checks whether recorded Euler angles match the gravity vector."
    )
    report.append(
        "Implied angles use Android device axes: "
        "pitch=atan2(-gx, hypot(gy,gz)), roll=atan2(gy, gz), degrees."
    )
    report.extend(stats_block("Recorded pitch (deg)", pitch))
    report.extend(stats_block("Recorded roll (deg)", roll))
    report.extend(stats_block("Gravity-implied pitch (deg)", implied_pitch))
    report.extend(stats_block("Gravity-implied roll (deg)", implied_roll))
    report.extend(stats_block("Gravity tilt from +Z (deg)", tilt_from_z))
    report.append(
        f"Pitch |recorded - implied|     : "
        f"mean={float(np.mean(pitch_err)):.3f} deg, "
        f"median={float(np.median(pitch_err)):.3f} deg"
    )
    report.append(
        f"Roll  |recorded - implied|     : "
        f"mean={float(np.mean(roll_err)):.3f} deg, "
        f"median={float(np.median(roll_err)):.3f} deg"
    )

    # 90 deg disagreement cannot be Euler wrap; it means a different convention or unused mapping.
    pitch_median = float(np.median(pitch_err))
    roll_median = float(np.median(roll_err))
    if pitch_median > 45.0 or roll_median > 45.0:
        warnings.append(
            "Recorded pitch/roll do not match gravity-implied attitude under the "
            "Android device-axis mapping. Gravity compensation itself does not use these Euler angles."
        )
        orientation_status = "WARNING"
    else:
        orientation_status = "PASSED"
    report.append(f"Pitch/roll vs gravity        : {orientation_status}")

    report.append("")
    report.append("-" * 60)
    report.append("CONCLUSIONS")
    report.append("-" * 60)
    if errors:
        overall = "FAIL"
    elif warnings:
        overall = "WARNING"
    else:
        overall = "PASS"

    report.append(f"Reconstruction (acc - grav)  : {'PASS' if reconstruction_ok and mag_ok else 'FAIL'}")
    report.append(f"Gravity magnitude            : {gravity_status.replace('PASSED', 'PASS').replace('FAILED', 'FAIL')}")
    report.append(f"Pitch/roll consistency       : {orientation_status.replace('PASSED', 'PASS')}")
    report.append(f"OVERALL                      : {overall}")
    if warnings:
        report.append("WARNINGS:")
        report.extend(f"  - {item}" for item in warnings)
    if errors:
        report.append("ERRORS:")
        report.extend(f"  - {item}" for item in errors)
    else:
        report.append(
            "Gravity compensation in imu_preprocessing.csv matches accelerometer minus gravity."
        )
    report.append("=" * 60)

    text = "\n".join(report) + "\n"
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {REPORT_FILE}")
    if errors:
        raise SystemExit(1)


def _write_and_print(report: list[str], errors: list[str], warnings: list[str]) -> None:
    report.append("OVERALL: FAIL")
    report.append("ERRORS:")
    report.extend(f"  - {item}" for item in errors)
    if warnings:
        report.append("WARNINGS:")
        report.extend(f"  - {item}" for item in warnings)
    text = "\n".join(report) + "\n"
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
