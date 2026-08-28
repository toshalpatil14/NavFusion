"""Run the tuned-Q/R EKF over the Day 2/3 navigation interface."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import MEASUREMENT_NOISE_R, PROCESS_NOISE_Q
from evaluate_ins_svw4 import align_with_source
from fusion_engine import FusionEngine
from models import NavigationInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "navigation_gnss_blackout.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ekf_trajectory.csv"


def main() -> None:
    navigation = pd.read_csv(INPUT_FILE)
    aligned = align_with_source(navigation)
    origin_lat, origin_lon = aligned.loc[0, ["source_gps_latitude", "source_gps_longitude"]]
    engine = FusionEngine(origin_lat, origin_lon, process_noise=PROCESS_NOISE_Q, measurement_noise=MEASUREMENT_NOISE_R)
    states = []
    for row in aligned.itertuples(index=False):
        available = bool(row.gnss_available)
        state = engine.update(NavigationInput(timestamp_ms=int(row.timestamp_ms), speed_mps=float(row.ai_speed_mps), heading_deg=float(row.heading_deg), gnss_available=available, gnss_lat=float(row.source_gps_latitude) if available else None, gnss_lon=float(row.source_gps_longitude) if available else None))
        states.append({"timestamp_ms": state.timestamp_ms, "dt_s": state.dt_s, "x_m": state.x_m, "y_m": state.y_m, "ekf_latitude": state.latitude, "ekf_longitude": state.longitude, "mode": state.mode})
    output = aligned.merge(pd.DataFrame(states), on="timestamp_ms", validate="one_to_one")
    output["process_noise_q"] = PROCESS_NOISE_Q
    output["measurement_noise_r"] = MEASUREMENT_NOISE_R
    if not np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("EKF output contains NaN or Inf.")
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"EKF samples: {len(output)}")
    print(f"Q={PROCESS_NOISE_Q}; R={MEASUREMENT_NOISE_R}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
