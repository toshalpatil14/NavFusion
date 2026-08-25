import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gnss_blackout.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "fusion_v1_trajectory.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("FUSION V1 - AI SPEED + HEADING NAVIGATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

required = [
    "timestamp_ms",
    "estimated_speed_kmh",
    "speed_mps",
    "heading_deg",
    "yaw_rate",
    "motion_state",
    "gnss_available",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n" +
        "\n".join(missing)
    )

print(f"\nInput rows: {len(df)}")


# ============================================================
# TIME
# ============================================================

df["dt"] = (
    df["timestamp_ms"].diff() / 1000.0
)

df.loc[0, "dt"] = 0.0

df["dt"] = df["dt"].clip(
    lower=0.0,
    upper=1.0
)


# ============================================================
# HEADING CONVENTION
# ============================================================

# Dataset phone heading is converted to the
# vehicle/navigation convention established during
# the previous heading-convention experiment.

heading = (
    df["heading_deg"].to_numpy()
    + 180.0
) % 360.0


# ============================================================
# SPEED
# ============================================================

speed = (
    df["speed_mps"]
    .to_numpy()
)

dt = (
    df["dt"]
    .to_numpy()
)


# ============================================================
# YAW-RATE INFORMATION
# ============================================================

yaw_rate = (
    df["yaw_rate"]
    .to_numpy()
)


# ============================================================
# FUSION V1
# ============================================================

# V1 uses:
#
#   AI speed       -> velocity magnitude
#   heading        -> absolute travel direction
#   yaw rate       -> monitored turn-rate information
#
# No GNSS position is used to propagate the trajectory.
#
# GNSS is retained only as an availability flag for
# the blackout experiment.


theta = np.deg2rad(
    heading
)

dx = (
    speed
    * np.sin(theta)
    * dt
)

dy = (
    speed
    * np.cos(theta)
    * dt
)


x = np.cumsum(dx)
y = np.cumsum(dy)


# ============================================================
# OUTPUT
# ============================================================

df["fusion_heading_deg"] = heading
df["dx_m"] = dx
df["dy_m"] = dy
df["x_m"] = x
df["y_m"] = y


output_columns = [
    "timestamp_ms",
    "estimated_speed_kmh",
    "speed_mps",
    "heading_deg",
    "fusion_heading_deg",
    "yaw_rate",
    "motion_state",
    "gnss_available",
    "dt",
    "dx_m",
    "dy_m",
    "x_m",
    "y_m",
]

df[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

distance = np.sum(
    np.sqrt(
        dx ** 2 +
        dy ** 2
    )
)

final_x = x[-1]
final_y = y[-1]

displacement = np.sqrt(
    final_x ** 2 +
    final_y ** 2
)

blackout = ~df["gnss_available"]

blackout_distance = np.sum(
    np.sqrt(
        dx[blackout.to_numpy()] ** 2 +
        dy[blackout.to_numpy()] ** 2
    )
)


print("\n" + "=" * 60)
print("FUSION V1 RESULTS")
print("=" * 60)

print(
    f"Samples: {len(df)}"
)

print(
    f"Total integrated distance: "
    f"{distance:.2f} m"
)

print(
    f"Final X: "
    f"{final_x:.2f} m"
)

print(
    f"Final Y: "
    f"{final_y:.2f} m"
)

print(
    f"Final displacement: "
    f"{displacement:.2f} m"
)

print(
    f"Blackout distance: "
    f"{blackout_distance:.2f} m"
)

print(
    f"GNSS available samples: "
    f"{df['gnss_available'].sum()}"
)

print(
    f"GNSS blackout samples: "
    f"{blackout.sum()}"
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)