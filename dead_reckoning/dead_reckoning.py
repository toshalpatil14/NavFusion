import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DEAD RECKONING BASELINE
# ============================================================

FILE_PATH = "data/processed_S1.csv"

print("Loading processed dataset...")

df = pd.read_csv(FILE_PATH)

print("Dataset loaded!")
print("Samples:", len(df))


# ============================================================
# 1. Extract data
# ============================================================

time = df["time_s"].to_numpy()

latitude = df["latitude"].to_numpy()
longitude = df["longitude"].to_numpy()

speed = df["gps_speed_ms"].to_numpy()

gyro_yaw = df["gyro_yaw"].to_numpy()

orientation_yaw = (
    df["orientation_yaw"].to_numpy()
)


# ============================================================
# 2. Time difference
# ============================================================

dt = np.diff(
    time,
    prepend=time[0]
)

dt = np.clip(
    dt,
    0,
    1.0
)


# ============================================================
# 3. Initial heading
# ============================================================

initial_heading = np.deg2rad(
    orientation_yaw[0]
)

heading = np.zeros(
    len(df)
)

heading[0] = initial_heading


# ============================================================
# 4. Integrate gyro yaw
# ============================================================

for i in range(1, len(df)):

    heading[i] = (
        heading[i - 1]
        + gyro_yaw[i] * dt[i]
    )

    heading[i] = np.arctan2(
        np.sin(heading[i]),
        np.cos(heading[i])
    )


# ============================================================
# 5. Distance travelled
# ============================================================

distance = (
    speed * dt
)


# ============================================================
# 6. Convert movement to North / East
# ============================================================

step_north = (
    distance * np.cos(heading)
)

step_east = (
    distance * np.sin(heading)
)


# ============================================================
# 7. Integrate local position
# ============================================================

estimated_north = np.cumsum(
    step_north
)

estimated_east = np.cumsum(
    step_east
)


# ============================================================
# 8. GPS -> local metres
# ============================================================

R = 6371000.0

lat0 = latitude[0]
lon0 = longitude[0]

lat_scale = (
    np.pi * R / 180.0
)

lon_scale = (
    np.pi
    * R
    * np.cos(np.deg2rad(lat0))
    / 180.0
)

gps_north = (
    latitude - lat0
) * lat_scale

gps_east = (
    longitude - lon0
) * lon_scale


# ============================================================
# 9. Position error
# ============================================================

position_error = np.sqrt(
    (estimated_north - gps_north) ** 2
    +
    (estimated_east - gps_east) ** 2
)


# ============================================================
# 10. Results
# ============================================================

print()
print("==============================")
print("DEAD RECKONING RESULTS")
print("==============================")

print(
    f"Initial error: "
    f"{position_error[0]:.2f} m"
)

print(
    f"Final error: "
    f"{position_error[-1]:.2f} m"
)

print(
    f"Mean error: "
    f"{np.mean(position_error):.2f} m"
)

print(
    f"Median error: "
    f"{np.median(position_error):.2f} m"
)

print(
    f"Maximum error: "
    f"{np.max(position_error):.2f} m"
)


# ============================================================
# 11. Convert DR position to GPS coordinates
# ============================================================

estimated_latitude = (
    lat0
    + estimated_north / lat_scale
)

estimated_longitude = (
    lon0
    + estimated_east / lon_scale
)


# ============================================================
# 12. Plot trajectory
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    longitude,
    latitude,
    label="GPS"
)

plt.plot(
    estimated_longitude,
    estimated_latitude,
    label="Dead reckoning"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "GPS vs Gyroscope Dead Reckoning"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 13. Plot error
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    position_error
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position error (m)")

plt.title(
    "Dead Reckoning Position Error"
)

plt.grid()

plt.tight_layout()

plt.show()