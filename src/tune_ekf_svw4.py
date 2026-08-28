"""Sweep EKF Q/R on the current S-Vw4 pipeline without changing production config."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import EARTH_RADIUS_M
from evaluate_ins_svw4 import align_with_source, metrics
from fusion_engine import FusionEngine
from models import NavigationInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "navigation_gnss_blackout.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ekf_tuning_svw4.csv"
SUMMARY_FILE = PROJECT_ROOT / "results" / "ekf_tuning_svw4.txt"
RECOVERY_WINDOW_MS = 10_000
PROCESS_NOISE_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
MEASUREMENT_NOISE_VALUES = [1.0, 4.0, 10.0, 25.0, 50.0, 100.0, 400.0, 800.0, 1600.0]


def run_ekf(aligned: pd.DataFrame, process_noise: float, measurement_noise: float) -> tuple[np.ndarray, np.ndarray]:
    origin_lat, origin_lon = aligned.loc[0, ["source_gps_latitude", "source_gps_longitude"]]
    engine = FusionEngine(origin_lat, origin_lon, process_noise=process_noise, measurement_noise=measurement_noise)
    x_m = np.empty(len(aligned), dtype=float)
    y_m = np.empty(len(aligned), dtype=float)
    for index, row in enumerate(aligned.itertuples(index=False)):
        available = bool(row.gnss_available)
        state = engine.update(NavigationInput(timestamp_ms=int(row.timestamp_ms), speed_mps=float(row.ai_speed_mps), heading_deg=float(row.heading_deg), gnss_available=available, gnss_lat=float(row.source_gps_latitude) if available else None, gnss_lon=float(row.source_gps_longitude) if available else None))
        x_m[index] = state.x_m
        y_m[index] = state.y_m
    if not (np.isfinite(x_m).all() and np.isfinite(y_m).all()):
        raise ValueError(f"EKF output contains NaN or Inf for Q={process_noise}, R={measurement_noise}.")
    return x_m, y_m


def score(aligned: pd.DataFrame, x_m: np.ndarray, y_m: np.ndarray, gps_x: np.ndarray, gps_y: np.ndarray, blackout: pd.Series, before: pd.Series, after: pd.Series, recovery: pd.Series) -> dict:
    error = np.hypot((x_m - x_m[0]) - gps_x, (y_m - y_m[0]) - gps_y)
    error_series = pd.Series(error, index=aligned.index)
    overall = metrics("OVERALL", error_series)
    before_metrics = metrics("BEFORE BLACKOUT", error_series.loc[before])
    blackout_metrics = metrics("DURING BLACKOUT", error_series.loc[blackout])
    after_metrics = metrics("AFTER BLACKOUT", error_series.loc[after])
    recovery_metrics = metrics("10S POST GNSS RECOVERY", error_series.loc[recovery])
    return {
        "process_noise_q": None,
        "measurement_noise_r": None,
        "overall_rmse_m": overall["rmse_m"],
        "overall_mean_m": overall["mean_error_m"],
        "overall_max_m": overall["max_error_m"],
        "recovery_10s_rmse_m": recovery_metrics["rmse_m"],
        "recovery_10s_mean_m": recovery_metrics["mean_error_m"],
        "recovery_10s_max_m": recovery_metrics["max_error_m"],
        "blackout_rmse_m": blackout_metrics["rmse_m"],
        "blackout_mean_m": blackout_metrics["mean_error_m"],
        "blackout_max_m": blackout_metrics["max_error_m"],
        "before_rmse_m": before_metrics["rmse_m"],
        "after_rmse_m": after_metrics["rmse_m"],
        "samples_overall": overall["samples"],
        "samples_blackout": blackout_metrics["samples"],
        "samples_recovery_10s": recovery_metrics["samples"],
        "max_gps_timestamp_difference_ms": int(aligned["gps_timestamp_difference_ms"].max()),
    }


def main() -> None:
    aligned = align_with_source(pd.read_csv(INPUT_FILE))
    lat0 = np.deg2rad(aligned["source_gps_latitude"].iloc[0])
    lon0 = np.deg2rad(aligned["source_gps_longitude"].iloc[0])
    gps_x = ((np.deg2rad(aligned["source_gps_longitude"]) - lon0) * EARTH_RADIUS_M * np.cos(lat0)).to_numpy(dtype=float)
    gps_y = ((np.deg2rad(aligned["source_gps_latitude"]) - lat0) * EARTH_RADIUS_M).to_numpy(dtype=float)
    blackout = ~aligned["gnss_available"].astype(bool)
    blackout_end_ms = int(aligned.loc[blackout, "timestamp_ms"].max())
    before = aligned["timestamp_ms"] < aligned.loc[blackout, "timestamp_ms"].min()
    after = aligned["timestamp_ms"] > blackout_end_ms
    recovery = after & (aligned["timestamp_ms"] <= blackout_end_ms + RECOVERY_WINDOW_MS)
    if not recovery.any():
        raise ValueError("No samples found in the 10-second post-GNSS-recovery window.")

    results = []
    total = len(PROCESS_NOISE_VALUES) * len(MEASUREMENT_NOISE_VALUES)
    test_number = 0
    print(f"Aligned S-Vw4 samples: {len(aligned)}")
    print(f"Configurations: {total}")
    print(f"Blackout samples: {int(blackout.sum())}; recovery samples: {int(recovery.sum())}")
    for process_noise in PROCESS_NOISE_VALUES:
        for measurement_noise in MEASUREMENT_NOISE_VALUES:
            test_number += 1
            print(f"[{test_number}/{total}] Q={process_noise} R={measurement_noise}")
            x_m, y_m = run_ekf(aligned, process_noise, measurement_noise)
            result = score(aligned, x_m, y_m, gps_x, gps_y, blackout, before, after, recovery)
            result["process_noise_q"] = process_noise
            result["measurement_noise_r"] = measurement_noise
            results.append(result)
            print(f"  overall RMSE={result['overall_rmse_m']:.4f} m; 10s recovery RMSE={result['recovery_10s_rmse_m']:.4f} m; blackout RMSE={result['blackout_rmse_m']:.4f} m")

    ranked = pd.DataFrame(results).sort_values(["overall_rmse_m", "recovery_10s_rmse_m", "blackout_rmse_m"], kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(OUTPUT_FILE, index=False)
    top = ranked.head(10)
    display_columns = ["rank", "process_noise_q", "measurement_noise_r", "overall_rmse_m", "recovery_10s_rmse_m", "blackout_rmse_m", "before_rmse_m", "after_rmse_m"]
    summary = [
        "EKF Q/R tuning on current S-Vw4 pipeline",
        "Reference: same-source S-Vw4 GPS only",
        "dt: timestamp_ms differences",
        "Heading: existing +180 deg correction applied once in FusionEngine",
        "Ranking: overall RMSE, then 10s post-GNSS-recovery RMSE, then blackout RMSE",
        f"Production config.py was not modified. Current production values remain Q=0.01, R=1600.0.",
        "",
        top[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        f"Saved all {len(ranked)} configurations to: {OUTPUT_FILE}",
    ]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
