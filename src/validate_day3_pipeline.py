"""Final structural validation for reproducible AI, INS, and EKF artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import MEASUREMENT_NOISE_R, PROCESS_NOISE_Q
from evaluate_ins_svw4 import source_gps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def check_frame(name: str, frame: pd.DataFrame, required: list[str], report: list[str], errors: list[str]) -> None:
    missing = [column for column in required if column not in frame]
    if missing:
        errors.append(f"{name}: missing columns {missing}")
        return
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all(): errors.append(f"{name}: NaN/Inf detected")
    if frame.timestamp_ms.duplicated().any(): errors.append(f"{name}: duplicate timestamps")
    if not frame.timestamp_ms.is_monotonic_increasing: errors.append(f"{name}: timestamps not ordered")
    dt = frame.timestamp_ms.diff().dropna() / 1000.0
    if (dt < 0).any(): errors.append(f"{name}: negative dt")
    report.append(f"{name}: rows={len(frame)}, dt_s min/median/max={dt.min():.3f}/{dt.median():.3f}/{dt.max():.3f}")


def main() -> None:
    errors, report = [], []
    ai = pd.read_csv(RESULTS / "ai_speed_output.csv")
    navigation = pd.read_csv(RESULTS / "navigation_gnss_blackout.csv")
    ins = pd.read_csv(RESULTS / "ins_trajectory.csv")
    ekf = pd.read_csv(RESULTS / "ekf_trajectory.csv")
    check_frame("AI", ai, ["timestamp_ms", "ai_speed_mps", "speed_confidence"], report, errors)
    check_frame("navigation", navigation, ["timestamp_ms", "ai_speed_mps", "heading_deg", "gnss_available"], report, errors)
    check_frame("INS", ins, ["timestamp_ms", "dt_s", "x_m", "y_m"], report, errors)
    check_frame("EKF", ekf, ["timestamp_ms", "dt_s", "x_m", "y_m", "process_noise_q", "measurement_noise_r"], report, errors)
    manifest = RESULTS / "ai_speed_output_manifest.txt"
    if not manifest.exists():
        errors.append("AI output reproducibility manifest is missing; run export_ai_speed_output.py after restoring the scaler.")
    elif "source_file=" + str(PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv") not in manifest.read_text(encoding="utf-8"):
        errors.append("AI output manifest does not identify the repository S-Vw4 source.")
    expected = source_gps()["gps_timestamp_ms"].iloc[49::10].to_numpy(dtype=np.int64)
    if not np.array_equal(ai.timestamp_ms.to_numpy(dtype=np.int64), expected): errors.append("AI timestamps do not follow source last-window-sample convention")
    report.append(f"AI timestamps: predictions={len(ai)}, first={ai.timestamp_ms.iloc[0]}, last={ai.timestamp_ms.iloc[-1]}, unique={not ai.timestamp_ms.duplicated().any()}")
    source_speed = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "IO-VNBD" / "S-Vw4" / "S-Vw4.csv", encoding="latin1"); source_speed.columns = source_speed.columns.str.strip()
    gps_speed = source_speed[["TIME SINCE START (ms)", "GPS SPEED (Kmh)"]].rename(columns={"TIME SINCE START (ms)": "timestamp_ms"}).sort_values("timestamp_ms")
    speed_join = pd.merge_asof(ai.sort_values("timestamp_ms"), gps_speed, on="timestamp_ms", direction="nearest", tolerance=60).dropna()
    speed_error = speed_join.ai_speed_mps - speed_join["GPS SPEED (Kmh)"] / 3.6
    report.append(f"AI/S-Vw4 speed units: matched={len(speed_join)}/{len(ai)}, MAE_mps={abs(speed_error).mean():.4f}, RMSE_mps={np.sqrt(np.mean(speed_error ** 2)):.4f}")
    blackout = ~navigation.gnss_available.astype(bool)
    report.append(f"blackout: samples={blackout.sum()}, first-to-last_s={(navigation.loc[blackout, 'timestamp_ms'].iloc[-1] - navigation.loc[blackout, 'timestamp_ms'].iloc[0]) / 1000:.3f}")
    if blackout.sum() != 200: errors.append("blackout is not exactly 200 samples")
    q, r = ekf.process_noise_q.unique(), ekf.measurement_noise_r.unique()
    report.append(f"EKF Q={q.tolist()}, R={r.tolist()}")
    if q.tolist() != [PROCESS_NOISE_Q] or r.tolist() != [MEASUREMENT_NOISE_R]: errors.append("EKF Q/R differ from required values")
    text = "\n".join(report + (["ERRORS:"] + errors if errors else ["VALIDATION: PASS"])) + "\n"
    (RESULTS / "day3_final_validation.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
