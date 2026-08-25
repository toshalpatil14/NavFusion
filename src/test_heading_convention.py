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

DR_FILE = (
    PROJECT_ROOT
    / "results"
    / "speed_heading_interface.csv"
)


# ============================================================
# LOAD
# ============================================================

gps = pd.read_csv(
    GPS_FILE,
    encoding="latin1"
)

gps.columns = gps.columns.str.strip()

gps = gps[
    [
        "TIME SINCE START (ms)",
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
).dropna()


dr = pd.read_csv(DR_FILE)

gps = gps.rename(
    columns={
        "TIME SINCE START (ms)": "timestamp_ms"
    }
)


# ============================================================
# MATCH TIMESTAMPS
# ============================================================

m = dr.merge(
    gps,
    on="timestamp_ms",
    how="inner"
)

print("Matched samples:", len(m))


# ============================================================
# GPS MOVEMENT HEADING
# ============================================================

R = 6371000.0

lat = np.deg2rad(
    m["GPS LATITUDE (degrees)"].to_numpy()
)

lon = np.deg2rad(
    m["GPS LONGITUDE (degrees)"].to_numpy()
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

distance = np.sqrt(
    dx ** 2 +
    dy ** 2
)

gps_heading = (
    np.degrees(
        np.arctan2(dx, dy)
    )
    + 360
) % 360


# ============================================================
# PHONE HEADING
# ============================================================

phone_heading = (
    m["heading_deg"]
    .to_numpy()[1:]
)


# Only use GPS movements >= 10 m.
valid = distance >= 10

gps_heading = gps_heading[valid]
phone_heading = phone_heading[valid]

print(
    "Valid movement samples:",
    len(gps_heading)
)


# ============================================================
# TEST TRANSFORMATIONS
# ============================================================

transformations = {

    "original":
        phone_heading,

    "plus_180":
        (phone_heading + 180) % 360,

    "mirror":
        (360 - phone_heading) % 360,

    "mirror_plus_180":
        (180 - phone_heading) % 360,
}


print()
print("=" * 70)
print("HEADING CONVENTION TEST")
print("=" * 70)

for name, transformed in transformations.items():

    error = np.abs(
        transformed -
        gps_heading
    )

    error = np.minimum(
        error,
        360 - error
    )

    print(
        f"{name:20s} "
        f"mean={error.mean():8.2f}°  "
        f"median={np.median(error):8.2f}°  "
        f"p90={np.percentile(error, 90):8.2f}°"
    )

print("=" * 70)