import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HEADING VALIDATION
# ============================================================

FILE_PATH = "data/processed_S1.csv"

print("Loading processed dataset...")

df = pd.read_csv(FILE_PATH)

print("Samples:", len(df))


# ============================================================
# 1. Extract data
# ============================================================

time = df["time_s"].to_numpy()

lat = df["latitude"].to_numpy()
lon = df["longitude"].to_numpy()

# Correct processed column name
speed = df["gps_speed_ms"].to_numpy()

orientation = df["orientation_yaw"].to_numpy()


# ============================================================
# 2. Convert GPS coordinates to local metres
# ============================================================

R = 6371000.0

lat0 = lat[0]
lon0 = lon[0]

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
    lat - lat0
) * lat_scale

east = (
    lon - lon0
) * lon_scale


# ============================================================
# 3. Calculate GPS movement heading
# ============================================================

WINDOW_SECONDS = 5.0

gps_heading = np.full(
    len(df),
    np.nan
)

gps_displacement = np.full(
    len(df),
    np.nan
)

for i in range(len(df)):

    target_time = (
        time[i] - WINDOW_SECONDS
    )

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

    gps_displacement[i] = distance

    # Require meaningful displacement
    if distance < 5.0:
        continue

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
# 4. Only use GPS heading while vehicle is moving
# ============================================================

moving = speed > 1.0

gps_heading[~moving] = np.nan


# ============================================================
# 5. Circular heading difference
# ============================================================

valid = np.isfinite(gps_heading)

def angular_difference(a, b):

    return np.abs(
        (a - b + 180.0) % 360.0 - 180.0
    )


heading_error = angular_difference(
    orientation[valid] % 360.0,
    gps_heading[valid]
)


# ============================================================
# 6. Statistics
# ============================================================

print()
print("==============================")
print("HEADING VALIDATION")
print("==============================")

print(
    f"Valid GPS heading samples: "
    f"{np.sum(valid)}"
)

print(
    f"Percentage valid: "
    f"{100 * np.mean(valid):.2f}%"
)

if np.sum(valid) > 0:

    print(
        f"Orientation vs GPS mean error: "
        f"{np.mean(heading_error):.2f} deg"
    )

    print(
        f"Orientation vs GPS median error: "
        f"{np.median(heading_error):.2f} deg"
    )

    print(
        f"Orientation vs GPS maximum error: "
        f"{np.max(heading_error):.2f} deg"
    )


# ============================================================
# 7. Plot heading comparison
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    orientation % 360,
    label="IMU orientation yaw"
)

plt.plot(
    time,
    gps_heading,
    ".",
    markersize=1,
    label="GPS movement heading"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")

plt.title(
    "IMU Orientation vs GPS Movement Heading"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 8. Plot heading error
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    time[valid],
    heading_error
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading error (degrees)")

plt.title(
    "IMU Orientation Heading Error"
)

plt.grid()

plt.tight_layout()

plt.show()
