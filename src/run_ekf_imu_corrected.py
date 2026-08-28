"""Run the production 4-state IMU + AI speed + GNSS EKF."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import EARTH_RADIUS_M, MEASUREMENT_NOISE_R, MODE_DEAD_RECKONING, MODE_GNSS_INS, PROCESS_NOISE_Q
    from coordinates import latlon_to_xy
except ImportError:
    from src.config import EARTH_RADIUS_M, MEASUREMENT_NOISE_R, MODE_DEAD_RECKONING, MODE_GNSS_INS, PROCESS_NOISE_Q
    from src.coordinates import latlon_to_xy

try:
    from ekf import PositionVelocityEKF
except ImportError:
    from src.ekf import PositionVelocityEKF

ROOT = Path(__file__).resolve().parents[1]
NAVIGATION_FILE = ROOT / "results" / "navigation_gnss_blackout.csv"
IMU_CORRECTION_FILE = ROOT / "results" / "imu_speed_correction.csv"
GPS_FILE = ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv"
OUTPUT_FILE = ROOT / "results" / "ekf_imu_corrected_trajectory.csv"
MAX_GPS_ALIGNMENT_MS = 60
EXPECTED_BLACKOUT_ROWS = 200
MAX_DT_S = 1.0


def require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def validate_timestamps(frame: pd.DataFrame, name: str) -> None:
    timestamps = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    if timestamps.isna().any() or not np.isfinite(timestamps).all():
        raise ValueError(f"{name} contains invalid timestamps")
    if timestamps.duplicated().any():
        raise ValueError(f"{name} contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps are not strictly increasing")


def math_heading(heading_deg: float) -> tuple[float, float]:
    travel = (heading_deg + 180.0) % 360.0
    return travel, math.radians(90.0 - travel)


def load_gnss() -> pd.DataFrame:
    source = pd.read_csv(GPS_FILE, encoding="latin1")
    source.columns = source.columns.str.strip()
    require_columns(source, ["TIME SINCE START (ms)", "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)"], "S-Vw4")
    gps = source[["TIME SINCE START (ms)", "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)"]].copy()
    gps.columns = ["timestamp_ms", "latitude_deg", "longitude_deg"]
    gps = gps.apply(pd.to_numeric, errors="coerce").dropna()
    gps["timestamp_ms"] = gps["timestamp_ms"].round().astype(np.int64)
    if gps["timestamp_ms"].duplicated().any():
        gps = gps.drop_duplicates("timestamp_ms", keep="first")
    gps = gps.sort_values("timestamp_ms").reset_index(drop=True)
    if len(gps) < 2:
        raise ValueError("S-Vw4 contains insufficient GPS samples")
    origin_lat = float(gps.iloc[0]["latitude_deg"])
    origin_lon = float(gps.iloc[0]["longitude_deg"])
    xy = gps.apply(
        lambda row: latlon_to_xy(float(row["latitude_deg"]), float(row["longitude_deg"]), origin_lat, origin_lon),
        axis=1,
    )
    gps["x_m"] = xy.map(lambda value: value[0])
    gps["y_m"] = xy.map(lambda value: value[1])
    return gps[["timestamp_ms", "x_m", "y_m"]]


def main() -> None:
    navigation = pd.read_csv(NAVIGATION_FILE)
    correction = pd.read_csv(IMU_CORRECTION_FILE)
    navigation.columns = navigation.columns.str.strip()
    correction.columns = correction.columns.str.strip()
    require_columns(navigation, ["timestamp_ms", "ai_speed_mps", "heading_deg", "gnss_available"], "navigation")
    require_columns(correction, ["timestamp_ms", "linear_ax", "linear_ay", "corrected_speed_mps"], "IMU correction")
    navigation["timestamp_ms"] = pd.to_numeric(navigation["timestamp_ms"], errors="coerce").round().astype("Int64")
    correction["timestamp_ms"] = pd.to_numeric(correction["timestamp_ms"], errors="coerce").round().astype("Int64")
    navigation = navigation.dropna(subset=["timestamp_ms"]).astype({"timestamp_ms": np.int64}).sort_values("timestamp_ms").reset_index(drop=True)
    correction = correction.dropna(subset=["timestamp_ms"]).astype({"timestamp_ms": np.int64}).sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)
    validate_timestamps(navigation, "navigation")
    if correction["timestamp_ms"].duplicated().any():
        raise ValueError("IMU correction contains duplicate timestamps")
    for column in ["ai_speed_mps", "heading_deg", "linear_ax", "linear_ay", "corrected_speed_mps"]:
        if column in navigation:
            navigation[column] = pd.to_numeric(navigation[column], errors="coerce")
    navigation["gnss_available"] = navigation["gnss_available"].map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
    )
    if navigation[["ai_speed_mps", "heading_deg"]].isna().any().any() or (navigation["ai_speed_mps"] < 0).any():
        raise ValueError("navigation contains invalid speed or heading values")
    blackout_rows = int((~navigation["gnss_available"]).sum())
    if blackout_rows != EXPECTED_BLACKOUT_ROWS:
        raise ValueError(f"Expected {EXPECTED_BLACKOUT_ROWS} blackout rows, found {blackout_rows}")
    correction["linear_ax"] = pd.to_numeric(correction["linear_ax"], errors="coerce")
    correction["linear_ay"] = pd.to_numeric(correction["linear_ay"], errors="coerce")
    correction["corrected_speed_mps"] = pd.to_numeric(correction["corrected_speed_mps"], errors="coerce")
    if not np.isfinite(correction[["linear_ax", "linear_ay"]]).all().all():
        raise ValueError("IMU acceleration contains NaN or Inf")
    if correction["corrected_speed_mps"].dropna().lt(0).any() or not np.isfinite(correction["corrected_speed_mps"].dropna()).all():
        raise ValueError("Corrected speed contains invalid values")
    merged = pd.merge_asof(navigation, correction[["timestamp_ms", "linear_ax", "linear_ay", "corrected_speed_mps"]], on="timestamp_ms", direction="backward")
    merged["speed_source"] = np.where(merged["corrected_speed_mps"].notna(), "IMU_CORRECTED_AI", "AI_FALLBACK")
    merged["ekf_input_speed_mps"] = merged["corrected_speed_mps"].fillna(merged["ai_speed_mps"])
    gnss = load_gnss()
    merged = pd.merge_asof(merged, gnss, on="timestamp_ms", direction="nearest", tolerance=MAX_GPS_ALIGNMENT_MS)
    if merged[["x_m", "y_m"]].isna().any().any():
        raise ValueError("S-Vw4 alignment failed within 60 ms")
    first = merged.iloc[0]
    initial_travel_heading, initial_theta = math_heading(float(first["heading_deg"]))
    initial_speed = float(first["ekf_input_speed_mps"])
    initial_travel_rad = math.radians(initial_travel_heading)
    ekf = PositionVelocityEKF(
        0.0,
        0.0,
        initial_speed * math.sin(initial_travel_rad),
        initial_speed * math.cos(initial_travel_rad),
        PROCESS_NOISE_Q,
        math.sqrt(MEASUREMENT_NOISE_R),
        1.0,
    )
    rows = []
    gnss_updates = 0
    previous_timestamp = None
    for row in merged.itertuples(index=False):
        timestamp = int(row.timestamp_ms)
        dt = 0.01 if previous_timestamp is None else (timestamp - previous_timestamp) / 1000.0
        if dt <= 0 or dt > MAX_DT_S:
            raise ValueError(f"Invalid dt={dt} at timestamp {timestamp}")
        previous_timestamp = timestamp
        travel_heading, theta = math_heading(float(row.heading_deg))
        travel_heading_rad = math.radians(travel_heading)
        speed = float(row.ekf_input_speed_mps)
        ax_body = 0.0 if pd.isna(row.linear_ax) else float(row.linear_ax)
        ay_body = 0.0 if pd.isna(row.linear_ay) else float(row.linear_ay)
        # Existing validated phone-to-vehicle map: forward=-linear_ay,
        # lateral=-linear_ax. Rotate vehicle axes into east/north.
        ax_world = -ay_body * math.cos(theta) - ax_body * math.sin(theta)
        ay_world = -ay_body * math.sin(theta) + ax_body * math.cos(theta)
        ekf.predict(ax_world, ay_world, dt)
        ekf.update_speed(speed, travel_heading_rad)
        used_gnss = bool(row.gnss_available)
        if used_gnss:
            ekf.update_gnss(float(row.x_m), float(row.y_m))
            gnss_updates += 1
        x, y, vx, vy = ekf.state()
        rows.append({
            "timestamp_ms": timestamp,
            "x_m": x,
            "y_m": y,
            "vx_mps": vx,
            "vy_mps": vy,
            "ekf_speed_mps": math.hypot(vx, vy),
            "heading_deg": float(row.heading_deg),
            "travel_heading_deg": travel_heading,
            "gnss_available": used_gnss,
            "dt_s": dt,
            "mode": MODE_GNSS_INS if used_gnss else MODE_DEAD_RECKONING,
            "speed_source": row.speed_source,
            "ai_speed_mps": float(row.ai_speed_mps),
            "corrected_speed_mps": float(row.corrected_speed_mps) if not pd.isna(row.corrected_speed_mps) else np.nan,
            "imu_accel_east_mps2": ax_world,
            "imu_accel_north_mps2": ay_world,
            "selected_speed_mps": speed,
            "speed_error_mps": math.hypot(vx, vy) - speed,
            "process_noise_q": PROCESS_NOISE_Q,
            "measurement_noise_r": MEASUREMENT_NOISE_R,
        })
    output = pd.DataFrame(rows)
    numeric = ["x_m", "y_m", "vx_mps", "vy_mps", "ekf_speed_mps", "dt_s"]
    if len(output) != len(navigation) or not np.isfinite(output[numeric]).all().all():
        raise ValueError("Output failed row-count or finite-state validation")
    output.to_csv(OUTPUT_FILE, index=False)
    fallback = int((output["speed_source"] == "AI_FALLBACK").sum())
    corrected = int((output["speed_source"] == "IMU_CORRECTED_AI").sum())
    blackout_output = output.loc[~output["gnss_available"]]
    final = blackout_output.iloc[-1]
    print(f"Output rows: {len(output)}")
    print(f"Blackout rows: {blackout_rows}")
    print(f"IMU rows used: {len(correction)}")
    print(f"Corrected-speed rows: {corrected}")
    print(f"Fallback-speed rows: {fallback}")
    print(f"GNSS updates disabled during blackout: {not bool(blackout_output['gnss_available'].any())}")
    print(f"Final blackout x/y: {final['x_m']:.4f}, {final['y_m']:.4f}")
    print(f"Final blackout vx/vy: {final['vx_mps']:.4f}, {final['vy_mps']:.4f}")
    print(f"Final blackout speed: {final['ekf_speed_mps']:.4f}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
