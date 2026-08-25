import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "clean_gps_reference.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum allowed consecutive GPS displacement.
# Earlier analysis found 969 jumps above this threshold.
MAX_JUMP_METERS = 100.0

EARTH_RADIUS_M = 6371000.0


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("CLEAN GPS REFERENCE CREATION")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1"
)

df.columns = df.columns.str.strip()

required = [
    "TIME SINCE START (ms)",
    "GPS LATITUDE (degrees)",
    "GPS LONGITUDE (degrees)",
    "GPS SPEED (Kmh)",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

df = df[required].copy()

for column in required:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna().reset_index(drop=True)

print(f"\nRows after numeric cleaning: {len(df)}")


# ============================================================
# CONSECUTIVE GPS DISPLACEMENT
# ============================================================

lat = np.deg2rad(
    df["GPS LATITUDE (degrees)"].to_numpy()
)

lon = np.deg2rad(
    df["GPS LONGITUDE (degrees)"].to_numpy()
)

dlat = np.diff(lat)
dlon = np.diff(lon)

mean_lat = (
    lat[:-1] + lat[1:]
) / 2.0

dx = (
    dlon
    * EARTH_RADIUS_M
    * np.cos(mean_lat)
)

dy = (
    dlat
    * EARTH_RADIUS_M
)

jump_distance = np.sqrt(
    dx ** 2 + dy ** 2
)

# First sample has no previous coordinate.
jump_distance = np.insert(
    jump_distance,
    0,
    0.0
)

df["gps_jump_m"] = jump_distance


# ============================================================
# VALIDITY FLAG
# ============================================================

df["gps_valid"] = (
    df["gps_jump_m"] <= MAX_JUMP_METERS
)


# Always retain the first sample.
df.loc[0, "gps_valid"] = True


# ============================================================
# SUMMARY
# ============================================================

rejected = (
    ~df["gps_valid"]
).sum()

valid = (
    df["gps_valid"]
).sum()


print("\n" + "=" * 60)
print("GPS CLEANING RESULTS")
print("=" * 60)

print(
    f"Input samples          : {len(df)}"
)

print(
    f"Valid GPS samples      : {valid}"
)

print(
    f"Rejected GPS samples   : {rejected}"
)

print(
    f"Rejection threshold    : "
    f"{MAX_JUMP_METERS:.1f} m"
)

print(
    f"Largest observed jump  : "
    f"{df['gps_jump_m'].max():.2f} m"
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "TIME SINCE START (ms)",
    "GPS LATITUDE (degrees)",
    "GPS LONGITUDE (degrees)",
    "GPS SPEED (Kmh)",
    "gps_jump_m",
    "gps_valid",
]

df[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)
print("CLEAN GPS REFERENCE COMPLETE")
print("=" * 60)