import pandas as pd
import numpy as np
from pathlib import Path

# ==========================
# PATHS
# ==========================

DATA_FILE = Path(
    r"F:\IO-VNBD-DATA\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset\S-Dataset\S-Vw4.csv"
)

OUTPUT_DIR = Path(r"F:\NavFusion\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# LOAD
# ==========================

df = pd.read_csv(DATA_FILE, encoding="latin1")
df.columns = df.columns.str.strip()

# ==========================
# REQUIRED COLUMNS
# ==========================

cols = [
    "TIME SINCE START (ms)",
    "GYROSCOPE Z (rad/s)",
    "ORIENTATION (Azimuth) (Â°)",
    "GPS SPEED (Kmh)"
]

df = df[cols].copy()

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().reset_index(drop=True)

# ==========================
# HEADING SMOOTHING
# ==========================

df["heading_deg"] = (
    df["ORIENTATION (Azimuth) (Â°)"]
    .rolling(window=5, center=True)
    .mean()
)

df["heading_deg"] = df["heading_deg"].bfill().ffill()

# ==========================
# YAW RATE
# ==========================

df["yaw_rate"] = df["GYROSCOPE Z (rad/s)"]

# ==========================
# MOTION STATE
# ==========================

states = []

for speed, yaw in zip(df["GPS SPEED (Kmh)"], df["yaw_rate"]):

    if speed < 1:
        states.append("STOP")

    elif yaw > 0.12:
        states.append("LEFT")

    elif yaw < -0.12:
        states.append("RIGHT")

    else:
        states.append("STRAIGHT")

df["motion_state"] = states

# ==========================
# SAVE
# ==========================

out = OUTPUT_DIR / "heading_motion_output.csv"
df.to_csv(out, index=False)

print("Saved:", out)
print(df.head())
print("\nMotion counts:")
print(df["motion_state"].value_counts())