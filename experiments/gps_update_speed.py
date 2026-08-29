import pandas as pd
import numpy as np

# ==================================================
# GPS UPDATE SPEED ANALYSIS
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


# ==================================================
# 2. Convert GPS coordinates to metres
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


# ==================================================
# 4. Calculate update-to-update speed
# ==================================================

update_times = time[update_indices]

update_lat = latitude[update_indices]
update_lon = longitude[update_indices]

update_north = north[update_indices]
update_east = east[update_indices]


gps_update_distance = np.sqrt(
    np.diff(update_north)**2
    +
    np.diff(update_east)**2
)

gps_update_time = np.diff(
    update_times
)

gps_update_speed = (
    gps_update_distance
    /
    gps_update_time
)


# ==================================================
# 5. Statistics
# ==================================================

print()
print("==============================")
print("GPS UPDATE SPEED ANALYSIS")
print("==============================")

print(
    f"GPS updates: "
    f"{len(update_indices)}"
)

print(
    f"Median update interval: "
    f"{np.median(gps_update_time):.2f} s"
)

print(
    f"Mean update interval: "
    f"{np.mean(gps_update_time):.2f} s"
)

print(
    f"Minimum update interval: "
    f"{np.min(gps_update_time):.2f} s"
)

print(
    f"Maximum update interval: "
    f"{np.max(gps_update_time):.2f} s"
)

print()

print(
    f"GPS update speed mean: "
    f"{np.mean(gps_update_speed):.2f} m/s"
)

print(
    f"GPS update speed median: "
    f"{np.median(gps_update_speed):.2f} m/s"
)

print(
    f"GPS update speed minimum: "
    f"{np.min(gps_update_speed):.2f} m/s"
)

print(
    f"GPS update speed maximum: "
    f"{np.max(gps_update_speed):.2f} m/s"
)


# ==================================================
# 6. Compare with sensor speed
# ==================================================

sensor_speed = df[
    "speed_ms"
].to_numpy()

sensor_speed_at_updates = (
    sensor_speed[update_indices]
)


# Need one fewer sample because
# GPS speed is between consecutive updates.

sensor_speed_compare = (
    sensor_speed_at_updates[1:]
)


valid = np.isfinite(
    gps_update_speed
) & np.isfinite(
    sensor_speed_compare
)


if np.sum(valid) > 0:

    error = (
        sensor_speed_compare[valid]
        -
        gps_update_speed[valid]
    )

    absolute_error = np.abs(error)

    print()
    print("==============================")
    print("UPDATE SPEED COMPARISON")
    print("==============================")

    print(
        f"Comparison samples: "
        f"{np.sum(valid)}"
    )

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
        sensor_speed_compare[valid],
        gps_update_speed[valid]
    )[0, 1]

    print(
        f"Correlation: "
        f"{correlation:.3f}"
    )
