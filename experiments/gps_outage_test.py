import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# GPS OUTAGE EXPERIMENT
# ==================================================

file_path = "data/processed_S1.csv"

print("Loading processed dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Samples:", len(df))


# ==================================================
# 1. Extract data
# ==================================================

time = df["time_s"].to_numpy()

gps_lat = df["latitude"].to_numpy()
gps_lon = df["longitude"].to_numpy()

speed = df["speed_ms"].to_numpy()
gyro_yaw = df["gyro_yaw"].to_numpy()

gps_accuracy = df["gps_accuracy"].to_numpy()


# ==================================================
# 2. Convert GPS to local metres
# ==================================================

R = 6371000.0

lat0 = gps_lat[0]
lon0 = gps_lon[0]

lat_scale = np.pi * R / 180.0

lon_scale = (
    np.pi
    * R
    * np.cos(np.deg2rad(lat0))
    / 180.0
)

gps_north = (
    gps_lat - lat0
) * lat_scale

gps_east = (
    gps_lon - lon0
) * lon_scale


# ==================================================
# 3. Time difference
# ==================================================

dt = np.diff(
    time,
    prepend=time[0]
)

dt = np.clip(
    dt,
    0,
    1.0
)


# ==================================================
# 4. Initial heading
# ==================================================

initial_heading = np.deg2rad(
    df["orientation_yaw"].iloc[0]
)

heading = np.zeros(
    len(df)
)

heading[0] = initial_heading


# ==================================================
# 5. Integrate gyroscope
# ==================================================

for i in range(1, len(df)):

    heading[i] = (
        heading[i - 1]
        + gyro_yaw[i] * dt[i]
    )

    heading[i] = np.arctan2(
        np.sin(heading[i]),
        np.cos(heading[i])
    )


# ==================================================
# 6. GPS outage definition
# ==================================================

# GPS will be considered unavailable
# between these two times.

OUTAGE_START = 2500.0
OUTAGE_END = 2800.0
gps_available = np.ones(
    len(df),
    dtype=bool
)

gps_available[
    (time >= OUTAGE_START)
    &
    (time <= OUTAGE_END)
] = False


print()
print("==============================")
print("GPS OUTAGE EXPERIMENT")
print("==============================")

print(
    f"GPS outage: "
    f"{OUTAGE_START:.0f} - "
    f"{OUTAGE_END:.0f} seconds"
)


# ==================================================
# 7. GPS + IMU navigation
# ==================================================

fused_north = np.zeros(
    len(df)
)

fused_east = np.zeros(
    len(df)
)

fused_north[0] = gps_north[0]
fused_east[0] = gps_east[0]


for i in range(1, len(df)):

    # ----------------------------------------------
    # IMU / speed prediction
    # ----------------------------------------------

    distance = (
        speed[i] * dt[i]
    )

    predicted_north = (
        fused_north[i - 1]
        + distance
        * np.cos(heading[i])
    )

    predicted_east = (
        fused_east[i - 1]
        + distance
        * np.sin(heading[i])
    )


    # ----------------------------------------------
    # GPS available
    # ----------------------------------------------

    if gps_available[i]:

        accuracy = max(
            float(gps_accuracy[i]),
            1.0
        )

        gps_weight = (
            1.0
            /
            (1.0 + accuracy / 10.0)
        )

        gps_weight = np.clip(
            gps_weight,
            0.05,
            0.8
        )

        fused_north[i] = (
            (1 - gps_weight)
            * predicted_north
            +
            gps_weight
            * gps_north[i]
        )

        fused_east[i] = (
            (1 - gps_weight)
            * predicted_east
            +
            gps_weight
            * gps_east[i]
        )


    # ----------------------------------------------
    # GPS unavailable
    # ----------------------------------------------

    else:

        # During GPS outage we rely entirely
        # on IMU + vehicle speed.

        fused_north[i] = predicted_north

        fused_east[i] = predicted_east


# ==================================================
# 8. Calculate position error
# ==================================================

position_error = np.sqrt(
    (fused_north - gps_north) ** 2
    +
    (fused_east - gps_east) ** 2
)


# ==================================================
# 9. Calculate outage metrics
# ==================================================

outage_mask = (
    ~gps_available
)

normal_mask = (
    gps_available
)


outage_error = (
    position_error[outage_mask]
)

normal_error = (
    position_error[normal_mask]
)


print()
print("==============================")
print("RESULTS")
print("==============================")


print(
    f"Normal GPS mean error: "
    f"{np.mean(normal_error):.2f} m"
)

print(
    f"Normal GPS maximum error: "
    f"{np.max(normal_error):.2f} m"
)


print(
    f"GPS outage mean error: "
    f"{np.mean(outage_error):.2f} m"
)

print(
    f"GPS outage maximum error: "
    f"{np.max(outage_error):.2f} m"
)


print(
    f"GPS outage final error: "
    f"{outage_error[-1]:.2f} m"
)


# ==================================================
# 10. Convert fused position to GPS coordinates
# ==================================================

fused_lat = (
    lat0
    + fused_north
    / lat_scale
)

fused_lon = (
    lon0
    + fused_east
    / lon_scale
)


# ==================================================
# 11. Plot trajectory
# ==================================================

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    gps_lon,
    gps_lat,
    label="GPS reference"
)

plt.plot(
    fused_lon,
    fused_lat,
    label="GPS + IMU fusion"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "GPS Outage — GPS vs GPS + IMU"
)

plt.legend()

plt.grid()

plt.show()


# ==================================================
# 12. Plot error
# ==================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    position_error,
    label="Position error"
)

plt.axvspan(
    OUTAGE_START,
    OUTAGE_END,
    alpha=0.2,
    label="GPS outage"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "GPS + IMU Position Error During GPS Outage"
)

plt.legend()

plt.grid()

plt.show()
