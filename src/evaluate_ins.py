"""Evaluate the Day 3 INS trajectory against timestamp-aligned vehicle GPS."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INS_FILE = PROJECT_ROOT / "results" / "ins_trajectory.csv"
VEHICLE_FILE = PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "V-Vw4.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv"
EVALUATION_FILE = PROJECT_ROOT / "results" / "ins_position_evaluation.csv"
METRICS_FILE = PROJECT_ROOT / "results" / "ins_metrics.txt"
EARTH_RADIUS_M = 6_371_000.0
TIMESTAMP_TOLERANCE_MS = 60


def vehicle_reference() -> pd.DataFrame:
    """Put the vehicle logger on the phone logger's elapsed timestamp clock."""
    vehicle = pd.read_csv(VEHICLE_FILE)
    vehicle.columns = vehicle.columns.str.strip()
    phone = pd.read_csv(PHONE_FILE, encoding="latin1")
    phone.columns = phone.columns.str.strip()
    required = [
        "Time Since Start of Day (seconds)",
        "Latitude (degrees)",
        "Longitude (degrees)",
        "Velocity (km/hr)",
    ]
    if missing := [column for column in required if column not in vehicle.columns]:
        raise ValueError("Vehicle reference missing columns: " + ", ".join(missing))
    vehicle = vehicle[required].apply(pd.to_numeric, errors="coerce").dropna().copy()
    vehicle["timestamp_ms"] = np.rint(
        phone["TIME SINCE START (ms)"].iloc[0]
        + (vehicle["Time Since Start of Day (seconds)"] - vehicle["Time Since Start of Day (seconds)"].iloc[0]) * 1000.0
    ).astype("int64")
    return vehicle.sort_values("timestamp_ms")


def metric_row(name: str, error: pd.Series) -> dict:
    values = error.to_numpy(dtype=float)
    return {
        "period": name,
        "samples": len(values),
        "mean_error_m": np.mean(values),
        "median_error_m": np.median(values),
        "rmse_m": np.sqrt(np.mean(values ** 2)),
        "p95_error_m": np.percentile(values, 95),
        "max_error_m": np.max(values),
    }


def main() -> None:
    ins = pd.read_csv(INS_FILE).sort_values("timestamp_ms")
    reference = vehicle_reference()
    merged = pd.merge_asof(
        ins,
        reference[["timestamp_ms", "Latitude (degrees)", "Longitude (degrees)"]],
        on="timestamp_ms",
        direction="nearest",
        tolerance=TIMESTAMP_TOLERANCE_MS,
    ).dropna(subset=["Latitude (degrees)", "Longitude (degrees)"]).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No INS samples matched vehicle GPS within 60 ms.")

    lat0 = np.deg2rad(merged["Latitude (degrees)"].iloc[0])
    lon0 = np.deg2rad(merged["Longitude (degrees)"].iloc[0])
    gps_x = (np.deg2rad(merged["Longitude (degrees)"]) - lon0) * EARTH_RADIUS_M * np.cos(lat0)
    gps_y = (np.deg2rad(merged["Latitude (degrees)"]) - lat0) * EARTH_RADIUS_M
    ins_x = merged["x_m"] - merged["x_m"].iloc[0]
    ins_y = merged["y_m"] - merged["y_m"].iloc[0]
    merged["gps_x_m"] = gps_x
    merged["gps_y_m"] = gps_y
    merged["ins_x_relative_m"] = ins_x
    merged["ins_y_relative_m"] = ins_y
    merged["position_error_m"] = np.hypot(ins_x - gps_x, ins_y - gps_y)

    blackout = ~merged["gnss_available"].astype(bool)
    before = merged["timestamp_ms"] < merged.loc[blackout, "timestamp_ms"].min()
    after = merged["timestamp_ms"] > merged.loc[blackout, "timestamp_ms"].max()
    metrics = pd.DataFrame([
        metric_row("BEFORE BLACKOUT", merged.loc[before, "position_error_m"]),
        metric_row("DURING BLACKOUT", merged.loc[blackout, "position_error_m"]),
        metric_row("AFTER BLACKOUT", merged.loc[after, "position_error_m"]),
        metric_row("OVERALL", merged["position_error_m"]),
    ])
    merged.to_csv(EVALUATION_FILE, index=False)
    METRICS_FILE.write_text(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n", encoding="utf-8")
    print(f"Matched vehicle GPS samples: {len(merged)}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved evaluation: {EVALUATION_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()
