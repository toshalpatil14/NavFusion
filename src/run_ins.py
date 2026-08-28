"""Run the first-stage 2D INS/dead-reckoning trajectory generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from ins import propagate_2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "navigation_gnss_blackout.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ins_trajectory.csv"


def main() -> None:
    trajectory = propagate_2d(pd.read_csv(INPUT_FILE))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(OUTPUT_FILE, index=False)
    print("2D INS/dead-reckoning complete")
    print(f"Samples: {len(trajectory)}")
    print(f"Integrated distance: {np.hypot(trajectory['dx_m'], trajectory['dy_m']).sum():.2f} m")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
