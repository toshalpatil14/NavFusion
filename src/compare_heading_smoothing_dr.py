import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "navigation_interface_10hz.csv"
)


df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

dt = (
    df["timestamp_ms"].diff()
    .fillna(0)
    / 1000.0
).clip(lower=0.0)

speed = df["speed_mps"].to_numpy()


# ------------------------------------------------------------
# Original heading
# ------------------------------------------------------------

heading_raw = (
    df["heading_deg"]
    .to_numpy()
)

# Dataset-specific phone-to-vehicle convention.
heading_raw = (
    heading_raw + 180.0
) % 360.0


# ------------------------------------------------------------
# Smoothed heading
# ------------------------------------------------------------

heading_phone_smoothed = (
    pd.Series(
        df["heading_deg"]
    )
    .rolling(
        window=5,
        center=True
    )
    .mean()
    .bfill()
    .ffill()
    .to_numpy()
)

heading_smooth = (
    heading_phone_smoothed + 180.0
) % 360.0


# ------------------------------------------------------------
# Integrator
# ------------------------------------------------------------

def integrate(heading_deg):

    theta = np.deg2rad(
        heading_deg
    )

    dx = (
        speed
        * np.sin(theta)
        * dt.to_numpy()
    )

    dy = (
        speed
        * np.cos(theta)
        * dt.to_numpy()
    )

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    distance = np.sum(
        np.sqrt(
            dx ** 2 +
            dy ** 2
        )
    )

    displacement = np.sqrt(
        x[-1] ** 2 +
        y[-1] ** 2
    )

    return (
        x[-1],
        y[-1],
        distance,
        displacement
    )


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

raw = integrate(
    heading_raw
)

smooth = integrate(
    heading_smooth
)


print("=" * 60)
print("HEADING SMOOTHING vs DR")
print("=" * 60)

print("\nRAW HEADING")
print(
    f"Distance       : {raw[2]:.2f} m"
)
print(
    f"Final X        : {raw[0]:.2f} m"
)
print(
    f"Final Y        : {raw[1]:.2f} m"
)
print(
    f"Displacement   : {raw[3]:.2f} m"
)

print("\n5-SAMPLE SMOOTHED HEADING")
print(
    f"Distance       : {smooth[2]:.2f} m"
)
print(
    f"Final X        : {smooth[0]:.2f} m"
)
print(
    f"Final Y        : {smooth[1]:.2f} m"
)
print(
    f"Displacement   : {smooth[3]:.2f} m"
)

print("\nDIFFERENCE")

print(
    f"Distance difference: "
    f"{smooth[2] - raw[2]:.2f} m"
)

print(
    f"Final X difference: "
    f"{smooth[0] - raw[0]:.2f} m"
)

print(
    f"Final Y difference: "
    f"{smooth[1] - raw[1]:.2f} m"
)

print(
    f"Displacement difference: "
    f"{smooth[3] - raw[3]:.2f} m"
)

print("=" * 60)