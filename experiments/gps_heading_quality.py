import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GPS HEADING QUALITY ANALYSIS
# ============================================================

FILE_PATH = "data/processed_S1.csv"

df = pd.read_csv(FILE_PATH)

print("Samples:", len(df))


# ============================================================
# 1. Data
# ============================================================

time = df["time_s"].to_numpy()

lat = df["latitude"].to_numpy()
lon = df["longitude"].to_numpy()

gps_speed = df["gps_speed_ms"].to_numpy()


# ============================================================
# 2. Convert GPS to local metres
# ============================================================

R = 6371000.0

lat0 = lat[0]

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
    lat - lat[0]
) * lat_scale

east = (
    lon - lon[0]
) * lon_scale


# ============================================================
# 3. Calculate GPS displacement over 5 seconds
# ============================================================

WINDOW = 5.0

gps_distance = np.full(
    len(df),
    np.nan
)

gps_heading = np.full(
    len(df),
    np.nan
)

for i in range(len(df)):

    target_time = time[i] - WINDOW

    j = np.searchsorted(
        time,
        target_time
    )

    if j >= i:
        continue

    dn = north[i] - north[j]
    de = east[i] - east[j]

    distance = np.sqrt(
        dn**2 + de**2
    )

    gps_distance[i] = distance

    if distance >= 5.0:

        gps_heading[i] = (
            np.degrees(
                np.arctan2(
                    de,
                    dn
                )
            )
            % 360
        )


# ============================================================
# 4. GPS position-derived speed
# ============================================================

gps_position_speed = (
    gps_distance / WINDOW
)


# ============================================================
# 5. Validity
# ============================================================

valid_heading = np.isfinite(
    gps_heading
)

valid_speed = np.isfinite(
    gps_position_speed
)


# ============================================================
# 6. Statistics
# ============================================================

print()
print("==============================")
print("GPS HEADING QUALITY")
print("==============================")

print(
    "Valid GPS heading samples:",
    np.sum(valid_heading)
)

print(
    "Total samples:",
    len(df)
)

print(
    "Percentage with reliable displacement:",
    f"{100 * np.mean(valid_heading):.2f}%"
)

print()

print(
    f"Dataset GPS speed mean: "
    f"{np.mean(gps_speed):.2f} m/s"
)

print(
    f"Dataset GPS speed median: "
    f"{np.median(gps_speed):.2f} m/s"
)

print()

# Only compare where position-derived speed exists
speed_error = (
    gps_position_speed[valid_speed]
    - gps_speed[valid_speed]
)

print(
    f"Position-derived GPS speed mean: "
    f"{np.mean(gps_position_speed[valid_speed]):.2f} m/s"
)

print(
    f"GPS position vs reported speed MAE: "
    f"{np.mean(np.abs(speed_error)):.2f} m/s"
)


# ============================================================
# 7. GPS displacement statistics
# ============================================================

print()

print(
    f"GPS displacement median "
    f"(5 sec): "
    f"{np.nanmedian(gps_distance):.2f} m"
)

print(
    f"GPS displacement mean "
    f"(5 sec): "
    f"{np.nanmean(gps_distance):.2f} m"
)

print(
    f"GPS displacement maximum "
    f"(5 sec): "
    f"{np.nanmax(gps_distance):.2f} m"
)


# ============================================================
# 8. Plot GPS displacement
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    gps_distance,
    label="GPS displacement (5 s)"
)

plt.axhline(
    5.0,
    linestyle="--",
    label="5 m threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Displacement (m)")

plt.title(
    "GPS Displacement Over 5 Seconds"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 9. Plot speed comparison
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    gps_speed,
    label="Dataset GPS speed"
)

plt.plot(
    time,
    gps_position_speed,
    label="Position-derived GPS speed"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (m/s)")

plt.title(
    "Dataset GPS Speed vs Position-Derived GPS Speed"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()
