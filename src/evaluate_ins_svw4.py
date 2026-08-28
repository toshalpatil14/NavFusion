"""Evaluate INS against GPS from the same S-Vw4 source stream."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INS_FILE = PROJECT_ROOT / "results" / "ins_trajectory.csv"
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ins_position_evaluation_svw4.csv"
METRICS_FILE = PROJECT_ROOT / "results" / "ins_metrics_svw4.txt"
MAX_ALIGNMENT_MS = 60
EARTH_RADIUS_M = 6_371_000.0


def source_gps() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_FILE, encoding="latin1")
    source.columns = source.columns.str.strip()
    columns = ["TIME SINCE START (ms)", "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)"]
    gps = source[columns].apply(pd.to_numeric, errors="coerce").dropna().copy()
    return gps.rename(columns={"TIME SINCE START (ms)": "gps_timestamp_ms", "GPS LATITUDE (degrees)": "source_gps_latitude", "GPS LONGITUDE (degrees)": "source_gps_longitude"}).sort_values("gps_timestamp_ms")


def align_with_source(trajectory: pd.DataFrame) -> pd.DataFrame:
    existing_reference_columns = ["gps_timestamp_ms", "source_gps_latitude", "source_gps_longitude", "gps_timestamp_difference_ms"]
    trajectory = trajectory.drop(columns=[column for column in existing_reference_columns if column in trajectory], errors="ignore")
    aligned = pd.merge_asof(trajectory.sort_values("timestamp_ms"), source_gps(), left_on="timestamp_ms", right_on="gps_timestamp_ms", direction="nearest", tolerance=MAX_ALIGNMENT_MS)
    aligned = aligned.dropna(subset=["source_gps_latitude", "source_gps_longitude"]).reset_index(drop=True)
    if len(aligned) != len(trajectory):
        raise ValueError("Every INS sample must align to S-Vw4 GPS within 60 ms.")
    aligned["gps_timestamp_difference_ms"] = (aligned["timestamp_ms"] - aligned["gps_timestamp_ms"]).abs()
    maximum = int(aligned["gps_timestamp_difference_ms"].max())
    if maximum > MAX_ALIGNMENT_MS:
        raise ValueError(f"GPS alignment difference {maximum} ms exceeds {MAX_ALIGNMENT_MS} ms.")
    return aligned


def metrics(name: str, error: pd.Series) -> dict:
    values = error.to_numpy(dtype=float)
    return {"period": name, "samples": len(values), "mean_error_m": np.mean(values), "median_error_m": np.median(values), "rmse_m": np.sqrt(np.mean(values ** 2)), "p95_error_m": np.percentile(values, 95), "max_error_m": np.max(values)}


def evaluate(trajectory: pd.DataFrame, x_column: str = "x_m", y_column: str = "y_m") -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_with_source(trajectory)
    lat0 = np.deg2rad(aligned["source_gps_latitude"].iloc[0])
    lon0 = np.deg2rad(aligned["source_gps_longitude"].iloc[0])
    gps_x = (np.deg2rad(aligned["source_gps_longitude"]) - lon0) * EARTH_RADIUS_M * np.cos(lat0)
    gps_y = (np.deg2rad(aligned["source_gps_latitude"]) - lat0) * EARTH_RADIUS_M
    estimated_x = aligned[x_column] - aligned[x_column].iloc[0]
    estimated_y = aligned[y_column] - aligned[y_column].iloc[0]
    aligned["gps_x_m"], aligned["gps_y_m"] = gps_x, gps_y
    aligned["estimated_x_relative_m"], aligned["estimated_y_relative_m"] = estimated_x, estimated_y
    aligned["position_error_m"] = np.hypot(estimated_x - gps_x, estimated_y - gps_y)
    blackout = ~aligned["gnss_available"].astype(bool)
    if not blackout.any():
        raise ValueError("No GNSS blackout samples found.")
    before = aligned["timestamp_ms"] < aligned.loc[blackout, "timestamp_ms"].min()
    after = aligned["timestamp_ms"] > aligned.loc[blackout, "timestamp_ms"].max()
    report = pd.DataFrame([metrics("BEFORE BLACKOUT", aligned.loc[before, "position_error_m"]), metrics("DURING BLACKOUT", aligned.loc[blackout, "position_error_m"]), metrics("AFTER BLACKOUT", aligned.loc[after, "position_error_m"]), metrics("OVERALL", aligned["position_error_m"])])
    blackout_data = aligned.loc[blackout]
    ins_straight = np.hypot(
        blackout_data["estimated_x_relative_m"].iloc[-1] - blackout_data["estimated_x_relative_m"].iloc[0],
        blackout_data["estimated_y_relative_m"].iloc[-1] - blackout_data["estimated_y_relative_m"].iloc[0],
    )
    gps_straight = np.hypot(
        blackout_data["gps_x_m"].iloc[-1] - blackout_data["gps_x_m"].iloc[0],
        blackout_data["gps_y_m"].iloc[-1] - blackout_data["gps_y_m"].iloc[0],
    )
    ins_path = np.hypot(
        np.diff(blackout_data["estimated_x_relative_m"]),
        np.diff(blackout_data["estimated_y_relative_m"]),
    ).sum()
    gps_path = np.hypot(
        np.diff(blackout_data["gps_x_m"]),
        np.diff(blackout_data["gps_y_m"]),
    ).sum()
    blackout_report = {
        "samples": len(blackout_data),
        "duration_s": (blackout_data["timestamp_ms"].iloc[-1] - blackout_data["timestamp_ms"].iloc[0]) / 1000.0,
        "ins_start": (blackout_data["estimated_x_relative_m"].iloc[0], blackout_data["estimated_y_relative_m"].iloc[0]),
        "ins_end": (blackout_data["estimated_x_relative_m"].iloc[-1], blackout_data["estimated_y_relative_m"].iloc[-1]),
        "reference_start": (blackout_data["gps_x_m"].iloc[0], blackout_data["gps_y_m"].iloc[0]),
        "reference_end": (blackout_data["gps_x_m"].iloc[-1], blackout_data["gps_y_m"].iloc[-1]),
        "ins_straight_m": ins_straight,
        "ins_path_m": ins_path,
        "gps_straight_m": gps_straight,
        "gps_path_m": gps_path,
        "mean_error_m": blackout_data["position_error_m"].mean(),
        "median_error_m": blackout_data["position_error_m"].median(),
        "rmse_m": np.sqrt(np.mean(blackout_data["position_error_m"] ** 2)),
        "p95_m": np.percentile(blackout_data["position_error_m"], 95),
        "max_error_m": blackout_data["position_error_m"].max(),
    }
    return aligned, report


def main() -> None:
    aligned, report = evaluate(pd.read_csv(INS_FILE))
    aligned.to_csv(OUTPUT_FILE, index=False)
    blackout_data = aligned.loc[~aligned["gnss_available"].astype(bool)]
    blackout_report = {
        "samples": len(blackout_data),
        "duration_s": (blackout_data["timestamp_ms"].iloc[-1] - blackout_data["timestamp_ms"].iloc[0]) / 1000.0,
        "ins_start": (blackout_data["estimated_x_relative_m"].iloc[0], blackout_data["estimated_y_relative_m"].iloc[0]),
        "ins_end": (blackout_data["estimated_x_relative_m"].iloc[-1], blackout_data["estimated_y_relative_m"].iloc[-1]),
        "reference_start": (blackout_data["gps_x_m"].iloc[0], blackout_data["gps_y_m"].iloc[0]),
        "reference_end": (blackout_data["gps_x_m"].iloc[-1], blackout_data["gps_y_m"].iloc[-1]),
        "ins_straight_m": np.hypot(
            blackout_data["estimated_x_relative_m"].iloc[-1] - blackout_data["estimated_x_relative_m"].iloc[0],
            blackout_data["estimated_y_relative_m"].iloc[-1] - blackout_data["estimated_y_relative_m"].iloc[0],
        ),
        "ins_path_m": np.hypot(np.diff(blackout_data["estimated_x_relative_m"]), np.diff(blackout_data["estimated_y_relative_m"])).sum(),
        "gps_straight_m": np.hypot(
            blackout_data["gps_x_m"].iloc[-1] - blackout_data["gps_x_m"].iloc[0],
            blackout_data["gps_y_m"].iloc[-1] - blackout_data["gps_y_m"].iloc[0],
        ),
        "gps_path_m": np.hypot(np.diff(blackout_data["gps_x_m"]), np.diff(blackout_data["gps_y_m"])).sum(),
        "mean_error_m": blackout_data["position_error_m"].mean(),
        "median_error_m": blackout_data["position_error_m"].median(),
        "rmse_m": np.sqrt(np.mean(blackout_data["position_error_m"] ** 2)),
        "p95_m": np.percentile(blackout_data["position_error_m"], 95),
        "max_error_m": blackout_data["position_error_m"].max(),
    }
    diagnostic = "\n".join([
        "",
        "BLACKOUT DRIFT (10-30 SECOND GNSS OUTAGE)",
        "Reference: S-Vw4 GPS coordinates.",
        "WARNING: approximately two large coordinate jumps are present in this S-Vw4 blackout reference; they are not normal vehicle motion.",
        f"Blackout samples: {blackout_report['samples']}",
        f"Blackout duration: {blackout_report['duration_s']:.4f} s",
        f"INS start position: ({blackout_report['ins_start'][0]:.4f}, {blackout_report['ins_start'][1]:.4f}) m",
        f"INS end position: ({blackout_report['ins_end'][0]:.4f}, {blackout_report['ins_end'][1]:.4f}) m",
        f"Reference start position: ({blackout_report['reference_start'][0]:.4f}, {blackout_report['reference_start'][1]:.4f}) m",
        f"Reference end position: ({blackout_report['reference_end'][0]:.4f}, {blackout_report['reference_end'][1]:.4f}) m",
        f"INS straight-line displacement: {blackout_report['ins_straight_m']:.4f} m",
        f"INS accumulated path length: {blackout_report['ins_path_m']:.4f} m",
        f"Reference straight-line displacement: {blackout_report['gps_straight_m']:.4f} m",
        f"Reference coordinate path length: {blackout_report['gps_path_m']:.4f} m",
        f"Mean position error: {blackout_report['mean_error_m']:.4f} m",
        f"Median error: {blackout_report['median_error_m']:.4f} m",
        f"RMSE: {blackout_report['rmse_m']:.4f} m",
        f"P95: {blackout_report['p95_m']:.4f} m",
        f"Maximum error: {blackout_report['max_error_m']:.4f} m",
    ])
    METRICS_FILE.write_text(f"Maximum GPS timestamp difference: {aligned['gps_timestamp_difference_ms'].max()} ms\n" + report.to_string(index=False, float_format=lambda value: f"{value:.4f}") + diagnostic + "\n", encoding="utf-8")
    print(f"Matched samples: {len(aligned)}; maximum GPS timestamp difference: {aligned['gps_timestamp_difference_ms'].max()} ms")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(diagnostic)


if __name__ == "__main__":
    main()
