"""Phone-frame IMU vectors -> vehicle and local east/north/up.

Orientation convention (S-Vw4, already used by ins.py, fusion_engine.py,
and imu_propagation.py):

    travel_heading_deg = (azimuth_deg + 180) % 360

    Compass: 0 = north, 90 = east, clockwise.
    East component  = magnitude * sin(travel_heading)
    North component = magnitude * cos(travel_heading)

Pitch/roll are NOT used. validate_gravity_compensation.py showed they do
not match the gravity vector (WARNING). Gravity is almost entirely along
phone +Z, so the phone XY plane is treated as the horizontal plane.

Axis map implied by azimuth + 180 (phone Y points opposite travel,
phone +Z is up):

    forward = -phone_Y
    lateral_right = -phone_X   # right-handed: forward x up
    up = +phone_Z

This script does not integrate velocity or position and does not call the EKF.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "imu_preprocessing.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "imu_frame_transform.csv"

EXPECTED_ROWS = 126526
REQUIRED_INPUT = [
    "timestamp_ms",
    "linear_ax",
    "linear_ay",
    "linear_az",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "azimuth_deg",
]
OUTPUT_COLUMNS = [
    "timestamp_ms",
    "linear_ax",
    "linear_ay",
    "linear_az",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "azimuth_deg",
    "travel_heading_deg",
    "accel_forward_mps2",
    "accel_lateral_mps2",
    "accel_up_mps2",
    "accel_east_mps2",
    "accel_north_mps2",
    "gyro_forward_radps",
    "gyro_lateral_radps",
    "gyro_up_radps",
]


def travel_heading_from_azimuth(azimuth_deg: np.ndarray) -> np.ndarray:
    return np.mod(azimuth_deg + 180.0, 360.0)


def phone_to_vehicle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return -y, -x, z


def vehicle_to_east_north(
    forward: np.ndarray,
    lateral: np.ndarray,
    travel_heading_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heading_rad = np.deg2rad(travel_heading_deg)
    sin_h = np.sin(heading_rad)
    cos_h = np.cos(heading_rad)
    east = forward * sin_h + lateral * cos_h
    north = forward * cos_h - lateral * sin_h
    return east, north


def main() -> None:
    print("=" * 60)
    print("IMU FRAME TRANSFORM")
    print("=" * 60)
    print("Input :", INPUT_FILE)
    print("Pitch/roll: NOT used (gravity-direction consistency WARNING).")
    print("Heading   : travel = (azimuth + 180) % 360  [existing S-Vw4 convention]")
    print()

    df = pd.read_csv(INPUT_FILE)
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    input_timestamps = df["timestamp_ms"].to_numpy().copy()
    work = df[REQUIRED_INPUT].copy()
    for column in REQUIRED_INPUT:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    ax = work["linear_ax"].to_numpy(dtype=float)
    ay = work["linear_ay"].to_numpy(dtype=float)
    az = work["linear_az"].to_numpy(dtype=float)
    gx = work["gyro_x"].to_numpy(dtype=float)
    gy = work["gyro_y"].to_numpy(dtype=float)
    gz = work["gyro_z"].to_numpy(dtype=float)
    azimuth = work["azimuth_deg"].to_numpy(dtype=float)

    azimuth = np.mod(azimuth, 360.0)
    travel = travel_heading_from_azimuth(azimuth)
    accel_fwd, accel_lat, accel_up = phone_to_vehicle(ax, ay, az)
    gyro_fwd, gyro_lat, gyro_up = phone_to_vehicle(gx, gy, gz)
    accel_east, accel_north = vehicle_to_east_north(accel_fwd, accel_lat, travel)

    out = pd.DataFrame(
        {
            "timestamp_ms": input_timestamps,
            "linear_ax": ax,
            "linear_ay": ay,
            "linear_az": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
            "azimuth_deg": azimuth,
            "travel_heading_deg": travel,
            "accel_forward_mps2": accel_fwd,
            "accel_lateral_mps2": accel_lat,
            "accel_up_mps2": accel_up,
            "accel_east_mps2": accel_east,
            "accel_north_mps2": accel_north,
            "gyro_forward_radps": gyro_fwd,
            "gyro_lateral_radps": gyro_lat,
            "gyro_up_radps": gyro_up,
        }
    )
    out = out[OUTPUT_COLUMNS]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    errors: list[str] = []
    warnings: list[str] = []
    if len(out) != EXPECTED_ROWS:
        errors.append(f"row count {len(out)} != {EXPECTED_ROWS}")
    if not np.array_equal(out["timestamp_ms"].to_numpy(), input_timestamps):
        errors.append("timestamps changed")
    if not out["timestamp_ms"].is_monotonic_increasing:
        errors.append("timestamps not ordered")
    if out["timestamp_ms"].duplicated().any():
        errors.append("duplicate timestamps")
    numeric = out.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("NaN/Inf in output")
    missing_out = [c for c in OUTPUT_COLUMNS if c not in out.columns]
    if missing_out:
        errors.append(f"missing output columns {missing_out}")

    heading_ok = (
        (out["travel_heading_deg"] >= 0.0).all()
        and (out["travel_heading_deg"] < 360.0).all()
        and (out["azimuth_deg"] >= 0.0).all()
        and (out["azimuth_deg"] < 360.0).all()
    )
    if not heading_ok:
        errors.append("heading outside [0, 360)")

    horiz_phone = np.hypot(ax, ay)
    horiz_vehicle = np.hypot(accel_fwd, accel_lat)
    horiz_enu = np.hypot(accel_east, accel_north)
    if not np.allclose(horiz_phone, horiz_vehicle, atol=1e-9):
        errors.append("vehicle horizontal accel magnitude != phone XY magnitude")
    if not np.allclose(horiz_phone, horiz_enu, atol=1e-9):
        errors.append("east/north accel magnitude != phone XY magnitude")
    if not np.allclose(accel_up, az, atol=1e-12):
        errors.append("accel_up is not phone linear_az")

    # Horizontal motion accel on this dataset is typically a few m/s^2; MEMS spikes can be larger.
    horiz_p99 = float(np.percentile(horiz_vehicle, 99))
    if horiz_p99 > 50.0:
        warnings.append(f"99th percentile horizontal accel is {horiz_p99:.2f} m/s^2")

    print("-" * 60)
    print("VALIDATION")
    print("-" * 60)
    print(f"Rows                         : {len(out)}")
    print(f"Timestamps unchanged         : {np.array_equal(out['timestamp_ms'].to_numpy(), input_timestamps)}")
    print(f"Timestamps ordered           : {bool(out['timestamp_ms'].is_monotonic_increasing)}")
    print(f"NaN/Inf                      : {'none' if np.isfinite(numeric).all() else 'FOUND'}")
    print(f"Required columns             : {len(OUTPUT_COLUMNS)} present")
    print(f"travel_heading in [0, 360)   : {heading_ok}")
    print(f"Horizontal |a| invariant     : {bool(np.allclose(horiz_phone, horiz_enu, atol=1e-9))}")
    print(f"accel_forward mean/std/p99   : {float(np.mean(accel_fwd)):.4f} / {float(np.std(accel_fwd)):.4f} / {float(np.percentile(accel_fwd, 99)):.4f} m/s^2")
    print(f"accel_lateral mean/std/p99   : {float(np.mean(accel_lat)):.4f} / {float(np.std(accel_lat)):.4f} / {float(np.percentile(accel_lat, 99)):.4f} m/s^2")
    print(f"accel_up mean/std            : {float(np.mean(accel_up)):.4f} / {float(np.std(accel_up)):.4f} m/s^2")
    print(f"Saved                        : {OUTPUT_FILE}")

    print()
    print("-" * 60)
    print("FRAME AND CONVENTION")
    print("-" * 60)
    print(
        "1. linear_ax/ay/az and gyro_x/y/z remain phone-sensor frame. "
        "accel_forward/lateral/up are vehicle (forward, right, up). "
        "accel_east/north are local navigation (same east/north as ins.py / fusion_engine.py). "
        "accel_up is world-up under the validated gravity=+Z assumption."
    )
    print(
        "2. Orientation: Android azimuth (0=N, 90=E) plus the existing S-Vw4 "
        "travel_heading = (azimuth + 180) % 360. Trig in radians."
    )
    print(
        "3. Pitch/roll were not used. Gravity-vs-Euler validation was WARNING; "
        "gravity lies along phone +Z, so only heading is applied in the XY plane."
    )
    print(
        "4. Consistent with S-Vw4: the same +180 travel heading and "
        "east=sin(heading), north=cos(heading) used by FusionEngine.predict and ins.propagate_2d."
    )

    if warnings:
        print()
        print("WARNINGS:")
        for item in warnings:
            print("  -", item)
    if errors:
        print()
        print("VALIDATION: FAIL")
        for item in errors:
            print("  -", item)
        raise SystemExit(1)

    print()
    print("VALIDATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
