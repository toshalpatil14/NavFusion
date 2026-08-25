import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "heading_motion_output.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("HEADING & MOTION MODULE V1")
print("=" * 60)

print("\nLoading:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE,
    encoding="latin1"
)

df.columns = df.columns.str.strip()

print(f"Rows loaded: {len(df)}")


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

def find_column(columns, keywords):

    for column in columns:

        column_lower = column.lower()

        if all(
            keyword.lower() in column_lower
            for keyword in keywords
        ):
            return column

    return None


TIME_COLUMN = find_column(
    df.columns,
    ["TIME SINCE START"]
)

GYRO_Z_COLUMN = find_column(
    df.columns,
    ["GYROSCOPE Z"]
)

AZIMUTH_COLUMN = find_column(
    df.columns,
    ["ORIENTATION (Azimuth)"]
)

GPS_SPEED_COLUMN = find_column(
    df.columns,
    ["GPS SPEED"]
)


required = {
    "timestamp": TIME_COLUMN,
    "gyro_z": GYRO_Z_COLUMN,
    "azimuth": AZIMUTH_COLUMN,
    "gps_speed": GPS_SPEED_COLUMN,
}

print("\nDetected columns:")

for name, column in required.items():

    print(f"{name}: {column}")

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:

    raise ValueError(
        "Could not find required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# SELECT DATA
# ============================================================

df = df[
    [
        TIME_COLUMN,
        GYRO_Z_COLUMN,
        AZIMUTH_COLUMN,
        GPS_SPEED_COLUMN,
    ]
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in df.columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna().reset_index(drop=True)

print(f"\nRows after cleaning: {len(df)}")


# ============================================================
# HEADING
# ============================================================

df["heading_deg"] = (
    df[AZIMUTH_COLUMN]
    .rolling(
        window=5,
        center=True
    )
    .mean()
)

df["heading_deg"] = (
    df["heading_deg"]
    .bfill()
    .ffill()
)

# Normalize heading to [0, 360)
df["heading_deg"] = (
    df["heading_deg"] % 360
)


# ============================================================
# YAW RATE
# ============================================================

df["yaw_rate"] = df[GYRO_Z_COLUMN]


# ============================================================
# MOTION STATE
# ============================================================

states = []

for speed, yaw in zip(
    df[GPS_SPEED_COLUMN],
    df["yaw_rate"]
):

    if speed < 1:

        states.append("STOP")

    elif yaw > 0.12:

        states.append("LEFT")

    elif yaw < -0.12:

        states.append("RIGHT")

    else:

        states.append("STRAIGHT")


df["motion_state"] = states


# ============================================================
# OUTPUT INTERFACE
# ============================================================

output = pd.DataFrame({

    "timestamp_ms":
        df[TIME_COLUMN].astype(np.int64),

    "heading_deg":
        df["heading_deg"].astype(np.float32),

    "yaw_rate":
        df["yaw_rate"].astype(np.float32),

    "motion_state":
        df["motion_state"],

})


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("HEADING & MOTION OUTPUT CREATED")
print("=" * 60)

print(f"Samples: {len(output)}")

print(
    f"First timestamp: "
    f"{output['timestamp_ms'].iloc[0]}"
)

print(
    f"Last timestamp: "
    f"{output['timestamp_ms'].iloc[-1]}"
)

print("\nMotion distribution:")

print(
    output["motion_state"]
    .value_counts()
)

print("\nFirst 5 rows:")

print(
    output.head().to_string(index=False)
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 60)