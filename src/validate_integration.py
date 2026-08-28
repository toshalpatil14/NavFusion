"""Validate Day 2/3 artifacts and report timestamp-aligned AI speed error."""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_ins import vehicle_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FILES = {
    "AI speed": RESULTS / "ai_speed_output.csv",
    "navigation interface": RESULTS / "navigation_interface_10hz.csv",
    "GNSS blackout": RESULTS / "navigation_gnss_blackout.csv",
    "fusion trajectory": RESULTS / "fusion_v1_trajectory.csv",
    "INS trajectory": RESULTS / "ins_trajectory.csv",
}
REQUIRED = {
    "AI speed": ["timestamp_ms", "ai_speed_mps", "speed_confidence"],
    "navigation interface": ["timestamp_ms", "ai_speed_mps", "speed_confidence", "heading_deg", "yaw_rate", "motion_state"],
    "GNSS blackout": ["timestamp_ms", "ai_speed_mps", "speed_confidence", "heading_deg", "yaw_rate", "motion_state", "gnss_available"],
    "fusion trajectory": ["timestamp_ms", "ai_speed_mps", "heading_deg", "yaw_rate", "gnss_available", "dt", "dt_s", "x_m", "y_m"],
    "INS trajectory": ["timestamp_ms", "ai_speed_mps", "heading_deg", "yaw_rate", "gnss_available", "dt_s", "x_m", "y_m"],
}


def main() -> None:
    errors, report = [], []
    frames = {}
    for name, path in FILES.items():
        if not path.exists():
            errors.append(f"Missing required file: {path}")
            continue
        frame = pd.read_csv(path)
        frames[name] = frame
        missing = [column for column in REQUIRED[name] if column not in frame]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
            continue
        numeric = frame.select_dtypes(include=[np.number])
        invalid = int((~np.isfinite(numeric.to_numpy())).sum())
        duplicate = int(frame["timestamp_ms"].duplicated().sum())
        ordered = bool(frame["timestamp_ms"].is_monotonic_increasing)
        dt = frame["timestamp_ms"].diff().dropna() / 1000.0
        report.append(f"{name}: rows={len(frame)}, invalid_numeric={invalid}, duplicate_timestamps={duplicate}, ordered={ordered}")
        report.append(f"{name} dt_s: min={dt.min():.3f}, median={dt.median():.3f}, mean={dt.mean():.3f}, max={dt.max():.3f}")
        if invalid or duplicate or not ordered:
            errors.append(f"{name}: invalid numeric data, duplicate timestamps, or timestamp ordering failure")

    blackout_frame = frames.get("GNSS blackout")
    if blackout_frame is not None and "gnss_available" in blackout_frame:
        blackout = ~blackout_frame["gnss_available"].astype(bool)
        count = int(blackout.sum())
        duration = (blackout_frame.loc[blackout, "timestamp_ms"].iloc[-1] - blackout_frame.loc[blackout, "timestamp_ms"].iloc[0]) / 1000.0
        report.append(f"blackout: samples={count}, first-to-last duration_s={duration:.3f}")
        if count != 200:
            errors.append(f"Blackout count is {count}; expected 200 samples.")

    ai = frames.get("AI speed")
    if ai is not None:
        reference = vehicle_reference()[["timestamp_ms", "Velocity (km/hr)"]]
        aligned = pd.merge_asof(ai.sort_values("timestamp_ms"), reference, on="timestamp_ms", direction="nearest", tolerance=60).dropna()
        speed_error = aligned["ai_speed_mps"] - aligned["Velocity (km/hr)"] / 3.6
        report.append(f"AI/reference timestamp alignment: matched={len(aligned)}/{len(ai)}, tolerance_ms=60")
        report.append(f"AI speed error (m/s): MAE={np.abs(speed_error).mean():.4f}, RMSE={np.sqrt(np.mean(speed_error ** 2)):.4f}")
        if len(aligned) != len(ai):
            errors.append("Some AI speeds did not align with vehicle timestamps.")

    output = "\n".join(report + (["ERRORS:"] + errors if errors else ["VALIDATION: PASS"])) + "\n"
    report_file = RESULTS / "integration_validation.txt"
    report_file.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved validation report: {report_file}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
