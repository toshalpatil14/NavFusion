import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# GPS INTERVAL SPEED VALIDATION
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

latitude = df["latitude"].to_numpy()
longitude = df["longitude"].to_numpy()

# GPS speed recorded in the original dataset,
# converted to metres/second during preprocessing.
sensor_speed = df["gps_speed_ms"].to_numpy()


# ==================================================
# 2. Convert GPS coordinates to local metres
# ==================================================

R = 6371000.0

lat0 = np.deg2rad(latitude[0])

north = (
    latitude - latitude[0]
) * np.pi * R / 180.0

east = (
    longitude - longitude[0]
) * np.pi * R * np.cos(lat0) / 180.0


# ==================================================
# 3. Find actual GPS position updates
# ==================================================

dn = np.diff(north)
de = np.diff(east)

distance = np.sqrt(
    dn**2 + de**2
)

changed = distance > 0

update_indices = (
    np.where(changed)[0] + 1
)

update_times = time[update_indices]


# ==================================================
# 4. Calculate GPS speed between updates
# ==================================================

gps_distance = np.sqrt(
    np.diff(north[update_indices])**2
    +
    np.diff(east[update_indices])**2
)

gps_interval = np.diff(update_times)

gps_speed = (
    gps_distance
    / gps_interval
)


# ==================================================
# 5. Calculate SENSOR AVERAGE speed
#    over the SAME GPS intervals
# ==================================================

sensor_interval_speed = []

valid_gps_speed = []

interval_times = []


for k in range(len(update_indices) - 1):

    start_idx = update_indices[k]
    end_idx = update_indices[k + 1]

    if end_idx <= start_idx:
        continue

    sensor_values = sensor_speed[
        start_idx:end_idx + 1
    ]

    valid_sensor = sensor_values[
        np.isfinite(sensor_values)
    ]

    if len(valid_sensor) == 0:
        continue

    sensor_mean = np.mean(
        valid_sensor
    )

    sensor_interval_speed.append(
        sensor_mean
    )

    valid_gps_speed.append(
        gps_speed[k]
    )

    interval_times.append(
        update_times[k + 1]
    )


sensor_interval_speed = np.array(
    sensor_interval_speed
)

valid_gps_speed = np.array(
    valid_gps_speed
)

interval_times = np.array(
    interval_times
)


# ==================================================
# 6. Statistics
# ==================================================

print()
print("==============================")
print("INTERVAL-MATCHED SPEED VALIDATION")
print("==============================")

print(
    f"GPS intervals: "
    f"{len(valid_gps_speed)}"
)

print(
    f"GPS speed mean: "
    f"{np.mean(valid_gps_speed):.2f} m/s"
)

print(
    f"GPS speed median: "
    f"{np.median(valid_gps_speed):.2f} m/s"
)

print(
    f"Dataset GPS speed mean: "
    f"{np.mean(sensor_interval_speed):.2f} m/s"
)

print(
    f"Dataset GPS speed median: "
    f"{np.median(sensor_interval_speed):.2f} m/s"
)


# ==================================================
# 7. Error analysis
# ==================================================

error = (
    sensor_interval_speed
    -
    valid_gps_speed
)

absolute_error = np.abs(
    error
)

print()
print("==============================")
print("SPEED ERROR")
print("==============================")

print(
    f"Mean absolute error: "
    f"{np.mean(absolute_error):.2f} m/s"
)

print(
    f"Median absolute error: "
    f"{np.median(absolute_error):.2f} m/s"
)

print(
    f"Maximum absolute error: "
    f"{np.max(absolute_error):.2f} m/s"
)

correlation = np.corrcoef(
    sensor_interval_speed,
    valid_gps_speed
)[0, 1]

print(
    f"Correlation: "
    f"{correlation:.3f}"
)


# ==================================================
# 8. Plot comparison
# ==================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    interval_times,
    sensor_interval_speed,
    label="Dataset GPS speed"
)

plt.plot(
    interval_times,
    valid_gps_speed,
    label="GPS position-derived speed"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Speed (m/s)"
)

plt.title(
    "Interval-Matched Dataset GPS Speed vs Position-Derived GPS Speed"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()