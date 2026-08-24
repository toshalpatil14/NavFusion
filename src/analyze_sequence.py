import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(
    r"D:\IDR-AI\data\raw\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset"
    r"\S-Dataset"
)

FILES = [
    "S-Vw4.csv",
    "S-S4.csv",
    "S-S2.csv",
    "S-M.csv",
    "S-Vfa02.csv",
    "S-Vw2.csv",
]

OUTPUT_DIR = Path(r"D:\IDR-AI\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_file(filename):

    path = DATA_DIR / filename

    df = pd.read_csv(
        path,
        encoding="latin1"
    )

    df.columns = df.columns.str.strip()

    return df


# --------------------------------------------------
# ANALYZE
# --------------------------------------------------

for filename in FILES:

    print("\n================================")
    print(filename)
    print("================================")

    df = load_file(filename)

    time = df["TIME SINCE START (ms)"] / 1000.0

    speed = df["GPS SPEED (Kmh)"]

    ax = df["ACCELEROMETER X (m/s²)"]
    ay = df["ACCELEROMETER Y (m/s²)"]
    az = df["ACCELEROMETER Z (m/s²)"]

    print("Samples:", len(df))
    print("Duration:", round(time.iloc[-1], 2), "seconds")
    print("Max speed:", speed.max(), "km/h")
    print("Mean speed:", round(speed.mean(), 2), "km/h")

    # -----------------------------
    # Speed plot
    # -----------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time, speed)

    plt.xlabel("Time (seconds)")
    plt.ylabel("GPS Speed (km/h)")
    plt.title(f"Speed - {filename}")

    plt.grid(True)

    plt.tight_layout()

    output = OUTPUT_DIR / f"{filename}_speed.png"

    plt.savefig(output)

    plt.close()

    # -----------------------------
    # Accelerometer plot
    # -----------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time, ax, label="Ax")
    plt.plot(time, ay, label="Ay")
    plt.plot(time, az, label="Az")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(f"Accelerometer - {filename}")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    output = OUTPUT_DIR / f"{filename}_accelerometer.png"

    plt.savefig(output)

    plt.close()

print("\n================================")
print("ANALYSIS COMPLETE")
print("================================")

print("Plots saved to:")
print(OUTPUT_DIR)