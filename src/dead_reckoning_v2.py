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
    / "navigation_interface_10hz.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_v2_trajectory.csv"
)

PLOT_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_v2_trajectory.png"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("2D DEAD RECKONING V2 - 10 HZ")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

required = [
    "timestamp_ms",
    "estimated_speed_kmh",
    "speed_mps",
    "heading_deg",
    "yaw_rate",
    "motion_state",
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

df.loc[0, "dt"] = 0.0

df["dt"] = df["dt"].clip(
    lower=0.0
)


# ============================================================
# HEADING CONVENTION
# ============================================================

# Phone azimuth is approximately 180 degrees
# opposite to vehicle movement heading for S-Vw4.

df["dr_heading_deg"] = (
    df["heading_deg"] + 180.0
) % 360.0


theta = np.deg2rad(
    df["dr_heading_deg"].to_numpy()
)

speed = (
    df["speed_mps"]
    .to_numpy()
)

dt = (
    df["dt"]
    .to_numpy()
)


# ============================================================
# VELOCITY COMPONENTS
# ============================================================

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


# ============================================================
# INTEGRATION
# ============================================================

x = np.cumsum(dx)
y = np.cumsum(dy)


df["dx_m"] = dx
df["dy_m"] = dy

df["x_m"] = x
df["y_m"] = y


# ============================================================
# DISTANCE
# ============================================================

step_distance = np.sqrt(
    dx ** 2 +
    dy ** 2
)

total_distance = np.sum(
    step_distance
)

final_x = x[-1]
final_y = y[-1]

final_displacement = np.sqrt(
    final_x ** 2 +
    final_y ** 2
)


# ============================================================
# SAVE
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

df[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DEAD RECKONING V2 RESULTS")
print("=" * 60)

print(
    f"Samples: "
    f"{len(df)}"
)

print(
    f"Total integrated distance: "
    f"{total_distance:.2f} m"
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
    f"{final_displacement:.2f} m"
)


# ============================================================
# MOTION DISTRIBUTION
# ============================================================

print("\nMotion distribution:")

print(
    df["motion_state"]
    .value_counts()
)


# ============================================================
# PLOT
# ============================================================

print("\nCreating trajectory plot...")

plt.figure(
    figsize=(10, 8)
)

plt.plot(
    df["x_m"],
    df["y_m"],
    label="DR V2"
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

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "2D Dead Reckoning V2 - 10 Hz - S-Vw4"
)

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
print("DEAD RECKONING V2 COMPLETE")
print("=" * 60)