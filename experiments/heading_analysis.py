import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# HEADING ANALYSIS
# ============================================================

FILE_PATH = "data/processed_S1.csv"

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

print("Samples:", len(df))

# ============================================================
# 1. Extract data
# ============================================================

time = df["time_s"].to_numpy()

lat = df["latitude"].to_numpy()
lon = df["longitude"].to_numpy()

speed = df["speed_ms"].to_numpy()

gyro_yaw = df["gyro_yaw"].to_numpy()

orientation_yaw = df["orientation_yaw"].to_numpy()

# ============================================================
# 2. Convert GPS coordinates to local metres
# ============================================================

R = 6371000.0

lat0 = lat[0]
lon0 = lon[0]

lat_scale = np.pi * R / 180.0

lon_scale = (
    np.pi
    * R
    * np.cos(np.deg2rad(lat0))
    / 180.0
)

north = (lat - lat0) * lat_scale
east = (lon - lon0) * lon_scale

# ============================================================
# 3. GPS movement heading
# ============================================================

# Use a 5-second displacement instead of
# consecutive GPS samples.
#
# This reduces the effect of GPS noise.

window_seconds = 5.0

gps_heading = np.full(len(df), np.nan)

for i in range(len(df)):

    target_time = time[i] - window_seconds

    j = np.searchsorted(time, target_time)

    if j >= i:
        continue

    dn = north[i] - north[j]
    de = east[i] - east[j]

    distance = np.sqrt(dn**2 + de**2)

    # Ignore extremely small movements.
    if distance < 2.0:
        continue

    # Heading convention:
    # 0 degrees = North
    # 90 degrees = East
    # 180 degrees = South
    # 270 degrees = West

    heading = np.degrees(
        np.arctan2(de, dn)
    )

    heading = heading % 360

    gps_heading[i] = heading

# ============================================================
# 4. Gyroscope-integrated heading
# ============================================================

dt = np.diff(
    time,
    prepend=time[0]
)

dt = np.clip(dt, 0, 1.0)

gyro_heading = np.zeros(len(df))

gyro_heading[0] = np.deg2rad(
    orientation_yaw[0]
)

for i in range(1, len(df)):

    gyro_heading[i] = (
        gyro_heading[i - 1]
        + gyro_yaw[i] * dt[i]
    )

    gyro_heading[i] = np.arctan2(
        np.sin(gyro_heading[i]),
        np.cos(gyro_heading[i])
    )

gyro_heading_deg = (
    np.degrees(gyro_heading) % 360
)

# ============================================================
# 5. Circular heading error
# ============================================================

valid = (
    np.isfinite(gps_heading)
)

def angular_difference(a, b):

    return np.abs(
        (a - b + 180) % 360 - 180
    )

orientation_error = angular_difference(
    orientation_yaw[valid],
    gps_heading[valid]
)

gyro_error = angular_difference(
    gyro_heading_deg[valid],
    gps_heading[valid]
)

# ============================================================
# 6. Statistics
# ============================================================

print()
print("==============================")
print("HEADING ANALYSIS")
print("==============================")

print(
    "Valid GPS heading samples:",
    np.sum(valid)
)

print()

print(
    f"Orientation vs GPS mean error: "
    f"{np.mean(orientation_error):.2f} deg"
)

print(
    f"Orientation vs GPS median error: "
    f"{np.median(orientation_error):.2f} deg"
)

print(
    f"Orientation vs GPS maximum error: "
    f"{np.max(orientation_error):.2f} deg"
)

print()

print(
    f"Gyro-integrated vs GPS mean error: "
    f"{np.mean(gyro_error):.2f} deg"
)

print(
    f"Gyro-integrated vs GPS median error: "
    f"{np.median(gyro_error):.2f} deg"
)

print(
    f"Gyro-integrated vs GPS maximum error: "
    f"{np.max(gyro_error):.2f} deg"
)

# ============================================================
# 7. Plot heading comparison
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    orientation_yaw,
    label="IMU orientation yaw"
)

plt.plot(
    time,
    gyro_heading_deg,
    label="Gyro-integrated heading"
)

plt.plot(
    time,
    gps_heading,
    ".",
    markersize=1,
    label="GPS movement heading (5 s)"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")

plt.title(
    "Heading Comparison: IMU, Gyroscope and GPS"
)

plt.legend()
plt.grid()

plt.show()

# ============================================================
# 8. Plot heading errors
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time[valid],
    orientation_error,
    label="Orientation yaw error"
)

plt.plot(
    time[valid],
    gyro_error,
    label="Gyro-integrated heading error"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading error (degrees)")

plt.title(
    "Heading Error Relative to GPS Movement"
)

plt.legend()
plt.grid()

plt.show()
