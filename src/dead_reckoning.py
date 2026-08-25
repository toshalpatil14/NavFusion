import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "speed_heading_interface.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_trajectory.csv"
)

PLOT_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_trajectory.png"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("2D DEAD RECKONING BASELINE")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

required = [
    "timestamp_ms",
    "speed_mps",
    "heading_deg",
]

missing = [
    col for col in required
    if col not in df.columns
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

# First sample has no previous timestamp.
df.loc[0, "dt"] = 0.0

# Protect against invalid/negative intervals.
df["dt"] = df["dt"].clip(lower=0.0)


# ============================================================
# HEADING → VELOCITY
# ============================================================

# Jayesh's convention:
# 0°   = North
# 90°  = East
# 180° = South
# 270° = West
#
# Therefore:
#
# East displacement  = v * sin(theta)
# North displacement = v * cos(theta)

dr_heading = (
    df["heading_deg"] + 180.0
) % 360.0

theta = np.deg2rad(
    dr_heading.values
)
speed = df["speed_mps"].values
dt = df["dt"].values


dx = (
    speed *
    np.sin(theta) *
    dt
)

dy = (
    speed *
    np.cos(theta) *
    dt
)


# ============================================================
# INTEGRATE
# ============================================================

x = np.cumsum(dx)
y = np.cumsum(dy)


df["dx_m"] = dx
df["dy_m"] = dy
df["dr_heading_deg"] = dr_heading

df["x_m"] = x
df["y_m"] = y


# ============================================================
# SAVE TRAJECTORY
# ============================================================

output_columns = [
    "timestamp_ms",
    "estimated_speed_kmh",
    "speed_mps",
    "heading_deg",
    "dr_heading_deg",
    "yaw_rate",
    "motion_state",
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

print("\n" + "=" * 60)
print("DEAD RECKONING RESULTS")
print("=" * 60)

print(f"Samples: {len(df)}")

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


# ============================================================
# PLOT
# ============================================================

print("\nCreating trajectory plot...")

plt.figure(figsize=(10, 8))

plt.plot(
    df["x_m"],
    df["y_m"]
)

plt.scatter(
    df["x_m"].iloc[0],
    df["y_m"].iloc[0],
    marker="o",
    label="Start"
)

plt.scatter(
    df["x_m"].iloc[-1],
    df["y_m"].iloc[-1],
    marker="x",
    label="End"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")
plt.title("2D Dead Reckoning Baseline - S-Vw4")

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nFiles saved:")

print(OUTPUT_FILE)
print(PLOT_FILE)

print("\n" + "=" * 60)
print("DEAD RECKONING BASELINE COMPLETE")
print("=" * 60)