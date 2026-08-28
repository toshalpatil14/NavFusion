"""Evaluate the EKF trajectory against same-source S-Vw4 GPS."""

from pathlib import Path

from evaluate_ins_svw4 import evaluate
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "ekf_imu_corrected_trajectory.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ekf_position_evaluation_svw4.csv"
METRICS_FILE = PROJECT_ROOT / "results" / "ekf_metrics_svw4.txt"


def main() -> None:
    evaluated, report = evaluate(pd.read_csv(INPUT_FILE))
    evaluated.to_csv(OUTPUT_FILE, index=False)
    q, r = evaluated.loc[0, ["process_noise_q", "measurement_noise_r"]]
    METRICS_FILE.write_text(f"Q={q}; R={r}\nMaximum GPS timestamp difference: {evaluated['gps_timestamp_difference_ms'].max()} ms\n" + report.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n", encoding="utf-8")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
