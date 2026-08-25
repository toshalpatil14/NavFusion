import pandas as pd
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = Path(
    r"F:\IO-VNBD-DATA\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset\S-Dataset\S-Vw4.csv"
)

OUTPUT_DIR = Path(r"F:\NavFusion\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(DATA_FILE, encoding="latin1")
df.columns = df.columns.str.strip()

# -----------------------------
# BUILD INTERFACE
# -----------------------------
out = pd.DataFrame()

out["timestamp_ms"] = pd.to_numeric(
    df["TIME SINCE START (ms)"], errors="coerce"
)

out["heading_deg"] = pd.to_numeric(
    df["ORIENTATION (Azimuth) (Â°)"], errors="coerce"
).bfill().ffill()

out["yaw_rate"] = pd.to_numeric(
    df["GYROSCOPE Z (rad/s)"], errors="coerce"
)

speed = pd.to_numeric(df["GPS SPEED (Kmh)"], errors="coerce")

def motion_state(yaw, spd):
    if spd < 1:
        return "STOP"
    if yaw > 0.12:
        return "LEFT"
    if yaw < -0.12:
        return "RIGHT"
    return "STRAIGHT"

out["motion_state"] = [
    motion_state(y, s)
    for y, s in zip(out["yaw_rate"], speed)
]

out = out.dropna().reset_index(drop=True)

# -----------------------------
# SAVE
# -----------------------------
save_path = OUTPUT_DIR / "heading_interface.csv"
out.to_csv(save_path, index=False)

print("Saved:", save_path)
print(out.head())