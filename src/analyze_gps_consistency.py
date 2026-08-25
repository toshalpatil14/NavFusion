import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)


df = pd.read_csv(
    GPS_FILE,
    encoding="latin1"
)

df.columns = df.columns.str.strip()

df = df[
    [
        "TIME SINCE START (ms)",
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
        "GPS SPEED (Kmh)",
    ]
].copy()

df = df.apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

# ------------------------------------------------------------
# GPS coordinate displacement
# ------------------------------------------------------------

R = 6371000.0

lat = np.deg2rad(
    df["GPS LATITUDE (degrees)"].to_numpy()
)

lon = np.deg2rad(
    df["GPS LONGITUDE (degrees)"].to_numpy()
)

dx = (
    np.diff(lon)
    * R
    * np.cos(lat[:-1])
)

dy = (
    np.diff(lat)
    * R
)

gps_step_m = np.sqrt(
    dx ** 2 +
    dy ** 2
)

dt = (
    np.diff(
        df["TIME SINCE START (ms)"].to_numpy()
    )
    / 1000.0
)

dt = np.maximum(dt, 0.0)

# ------------------------------------------------------------
# Coordinate-derived speed
# ------------------------------------------------------------

coordinate_speed_kmh = np.zeros(
    len(df)
)

valid_dt = dt > 0

coordinate_speed_kmh[1:][valid_dt] = (
    gps_step_m[valid_dt]
    / dt[valid_dt]
    * 3.6
)

# ------------------------------------------------------------
# GPS-speed-derived distance
# ------------------------------------------------------------

speed = (
    df["GPS SPEED (Kmh)"]
    .to_numpy()
)

speed_distance = np.sum(
    speed[:-1]
    / 3.6
    * dt
)

coordinate_distance = np.sum(
    gps_step_m
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 60)
print("GPS CONSISTENCY ANALYSIS")
print("=" * 60)

print(
    f"Samples: {len(df)}"
)

print(
    f"Duration: "
    f"{df['TIME SINCE START (ms)'].iloc[-1] - df['TIME SINCE START (ms)'].iloc[0]}"
    f" ms"
)

print()

print(
    f"Distance from GPS speed : "
    f"{speed_distance:.2f} m"
)

print(
    f"Distance from GPS coords: "
    f"{coordinate_distance:.2f} m"
)

print()

print(
    f"Coordinate/GPS-speed ratio: "
    f"{coordinate_distance / speed_distance:.2f}"
)

print()

print("Coordinate-derived speed:")

print(
    pd.Series(
        coordinate_speed_kmh
    ).describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
            0.999,
        ]
    ).to_string()
)

print()

print("Recorded GPS speed:")

print(
    pd.Series(speed).describe().to_string()
)

print()

# ------------------------------------------------------------
# How many coordinate jumps imply impossible speeds?
# ------------------------------------------------------------

for threshold in [
    50,
    100,
    200,
    500,
    1000,
]:

    count = np.sum(
        coordinate_speed_kmh >
        threshold
    )

    print(
        f"Coordinate speed > "
        f"{threshold:4d} km/h: "
        f"{count}"
    )

print("=" * 60)