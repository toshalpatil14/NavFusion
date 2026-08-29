import pandas as pd
import matplotlib.pyplot as plt

file_path = (
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "S (Driver A)/S1/S-S1.csv"
)

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

time = df["TIME SINCE START (ms)"] / 1000
# -----------------------------
# Accelerometer
# -----------------------------

plt.figure(figsize=(12, 6))

plt.plot(time, df["ACCELEROMETER X (m/s²)"], label="X")
plt.plot(time, df["ACCELEROMETER Y (m/s²)"], label="Y")
plt.plot(time, df["ACCELEROMETER Z (m/s²)"], label="Z")

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("S1 Smartphone Accelerometer")

plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# Gyroscope
# -----------------------------

plt.figure(figsize=(12, 6))

plt.plot(time, df["GYROSCOPE Yaw (rad/s)"], label="Yaw")
plt.plot(time, df["GYROSCOPE Pitch (rad/s)"], label="Pitch")
plt.plot(time, df["GYROSCOPE Roll (rad/s)"], label="Roll")

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("S1 Smartphone Gyroscope")

plt.legend()
plt.grid(True)

plt.show()