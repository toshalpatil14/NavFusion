import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# SPEED VALIDATION — 5 SECOND GPS WINDOW
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

sensor_speed = df["speed_ms"].to_numpy()


# ==================================================
# 2. GPS coordinates -> local metres
# ==================================================

R = 6371000.0

lat0 = latitude[0]

lat_scale = (
    np.pi * R / 180.0
)

lon_scale = (
    np.pi
    * R
    * np.cos(np.deg2rad(lat0))
    / 180.0
)

north = (
    latitude - latitude[0]
) * lat_scale

east = (
    longitude - longitude[0]
) * lon_scale


# ==================================================
# 3. 5-second displacement window
# ==================================================

WINDOW = 5.0

gps_speed = np.full(
    len(df),
    np.nan
)

gps_displacement = np.full(
    len(df),
    np.nan
)


for i in range(len(df)):

    target_time = time[i] - WINDOW

    j = np.searchsorted(
        time,
        target_time
    )

    if j < 0:
        continue

    if j >= len(df):
        continue

    elapsed = (
        time[i] - time[j]
    )

    if elapsed <= 0:
        continue

    dn = (
        north[i] - north[j]
    )

    de = (
        east[i] - east[j]
    )

    displacement = np.sqrt(
        dn**2 + de**2
    )

    gps_displacement[i] = displacement

    gps_speed[i] = (
        displacement
        / elapsed
    )


# ==================================================
# 4. Reliability filtering
# ==================================================

# A 5-second displacement smaller than 5 m
# is considered unreliable for this experiment.

RELIABILITY_THRESHOLD = 5.0

reliable = (
    np.isfinite(gps_displacement)
    &
    (
        gps_displacement
        >=
        RELIABILITY_THRESHOLD
    )
)


gps_speed_reliable = gps_speed.copy()

gps_speed_reliable[
    ~reliable
] = np.nan


# ==================================================
# 5. Remove unrealistic GPS speed
# ==================================================

gps_speed_reliable[
    gps_speed_reliable > 60
] = np.nan


# ==================================================
# 6. Statistics
# ==================================================

print()
print("==============================")
print("SPEED VALIDATION")
print("==============================")


print(
    f"Sensor speed mean: "
    f"{np.nanmean(sensor_speed):.2f} m/s"
)

print(
    f"Sensor speed median: "
    f"{np.nanmedian(sensor_speed):.2f} m/s"
)

print(
    f"Sensor speed minimum: "
    f"{np.nanmin(sensor_speed):.2f} m/s"
)

print(
    f"Sensor speed maximum: "
    f"{np.nanmax(sensor_speed):.2f} m/s"
)


print()

print(
    f"GPS 5-second speed mean: "
    f"{np.nanmean(gps_speed_reliable):.2f} m/s"
)

print(
    f"GPS 5-second speed median: "
    f"{np.nanmedian(gps_speed_reliable):.2f} m/s"
)

print(
    f"GPS 5-second speed minimum: "
    f"{np.nanmin(gps_speed_reliable):.2f} m/s"
)

print(
    f"GPS 5-second speed maximum: "
    f"{np.nanmax(gps_speed_reliable):.2f} m/s"
)


# ==================================================
# 7. Reliability percentage
# ==================================================

reliable_percentage = (
    np.sum(reliable)
    /
    len(df)
    *
    100
)

print()

print(
    f"Reliable GPS speed samples: "
    f"{np.sum(reliable)}"
)

print(
    f"Percentage with reliable GPS speed: "
    f"{reliable_percentage:.2f}%"
)


# ==================================================
# 8. Compare sensor and GPS speed
# ==================================================

comparison = (
    reliable
    &
    np.isfinite(sensor_speed)
    &
    np.isfinite(gps_speed_reliable)
)

if np.sum(comparison) > 0:

    speed_error = (
        sensor_speed[comparison]
        -
        gps_speed_reliable[comparison]
    )

    absolute_error = np.abs(
        speed_error
    )

    print()
    print("==============================")
    print("SPEED ERROR")
    print("==============================")

    print(
        f"Comparison samples: "
        f"{np.sum(comparison)}"
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
        sensor_speed[comparison],
        gps_speed_reliable[comparison]
    )[0, 1]

    print(
        f"Correlation: "
        f"{correlation:.3f}"
    )


# ==================================================
# 9. Plot speed comparison
# ==================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    sensor_speed,
    label="Sensor speed"
)

plt.plot(
    time,
    gps_speed_reliable,
    label="GPS-derived speed (5 s)"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Speed (m/s)"
)

plt.title(
    "Sensor Speed vs GPS-Derived Speed"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ==================================================
# 10. Plot GPS displacement
# ==================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    gps_displacement,
    label="GPS displacement (5 s)"
)

plt.axhline(
    RELIABILITY_THRESHOLD,
    linestyle="--",
    label="5 m reliability threshold"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Displacement (m)"
)

plt.title(
    "GPS Displacement Over 5 Seconds"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()