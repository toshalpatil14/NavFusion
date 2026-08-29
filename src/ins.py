"""Small, explicit 2D INS/dead-reckoning building blocks.

The heading module supplies a measured phone azimuth at every navigation
sample. Its established S-Vw4 convention is 180 degrees opposite vehicle
travel. We use that measured heading directly rather than integrating yaw
rate a second time; yaw rate remains in the trajectory for diagnostics.
"""

import numpy as np
import pandas as pd

REQUIRED_INPUT_COLUMNS = [
    "timestamp_ms",
    "ai_speed_mps",
    "heading_deg",
    "yaw_rate",
]


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    result = frame.copy()
    for column in REQUIRED_INPUT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[REQUIRED_INPUT_COLUMNS].isna().any().any():
        raise ValueError("INS input contains non-numeric or missing required values.")
    if not np.isfinite(result[REQUIRED_INPUT_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("INS input contains NaN or Inf.")
    if result["timestamp_ms"].duplicated().any():
        raise ValueError("INS input contains duplicate timestamps.")
    if not result["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("INS input timestamps must be strictly increasing.")
    return result


def propagate_2d(frame: pd.DataFrame) -> pd.DataFrame:
    """Propagate east/north position from AI speed and measured heading.

    ``dx = speed * cos(theta) * dt`` and ``dy = speed * sin(theta) * dt``.
    Theta is the compass heading converted to a mathematical east-zero angle.
    """
    result = _validate_input(frame)
    dt = result["timestamp_ms"].diff().fillna(0.0).to_numpy(dtype=float) / 1000.0
    travel_heading_deg = (result["heading_deg"].to_numpy(dtype=float) + 180.0) % 360.0
    theta_rad = np.deg2rad(90.0 - travel_heading_deg)
    speed_mps = result["ai_speed_mps"].to_numpy(dtype=float)
    dx = speed_mps * np.cos(theta_rad) * dt
    dy = speed_mps * np.sin(theta_rad) * dt

    result["dt_s"] = dt
    result["ins_heading_deg"] = travel_heading_deg
    result["heading_math_rad"] = theta_rad
    result["dx_m"] = dx
    result["dy_m"] = dy
    result["x_m"] = np.cumsum(dx)
    result["y_m"] = np.cumsum(dy)
    return result
