import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates in meters.
    """

    R = 6371000.0  # Earth radius in meters

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


# ============================================================
# GPS + IMU SENSOR FUSION
# STEP 10 — GPS CLEANING
# ============================================================

FILE_PATH = "data/processed_S1.csv"
GPS_SPEED_THRESHOLD = 15.0

print("Loading processed dataset...")

df = pd.read_csv(FILE_PATH)

print("Dataset loaded!")
print("Samples:", len(df))


# ============================================================
# 1. Extract GPS data
# ============================================================

time = df["time_s"].to_numpy(dtype=float)
gps_lat = df["latitude"].to_numpy(dtype=float)
gps_lon = df["longitude"].to_numpy(dtype=float)
gps_accuracy = df["gps_accuracy"].to_numpy(dtype=float)


# ============================================================
# 2. Identify actual GPS update samples
# ============================================================
# The processed dataset contains IMU samples at about 10 Hz, while
# GPS coordinates remain unchanged for many rows. Therefore GPS speed
# must be calculated between actual GPS coordinate updates, not between
# every sensor row.

gps_changed = (
    (gps_lat[1:] != gps_lat[:-1])
    | (gps_lon[1:] != gps_lon[:-1])
)

gps_update_indices = np.concatenate(
    ([0], np.where(gps_changed)[0] + 1)
)

gps_update_time = time[gps_update_indices]
gps_update_lat = gps_lat[gps_update_indices]
gps_update_lon = gps_lon[gps_update_indices]


# ============================================================
# 3. Calculate GPS movement between actual GPS updates
# ============================================================

gps_update_distance = np.zeros(
    len(gps_update_indices),
    dtype=float
)

gps_update_distance[1:] = np.array([
    haversine_distance(
        gps_update_lat[i - 1],
        gps_update_lon[i - 1],
        gps_update_lat[i],
        gps_update_lon[i]
    )
    for i in range(1, len(gps_update_indices))
])

gps_update_dt = np.diff(
    gps_update_time,
    prepend=gps_update_time[0]
)
gps_update_dt[0] = 1.0
gps_update_dt = np.maximum(
    gps_update_dt,
    1e-6
)

gps_update_speed = (
    gps_update_distance / gps_update_dt
)
gps_update_speed[0] = 0.0


# ============================================================
# 4. Detect suspicious GPS updates
# ============================================================

gps_update_outlier = (
    gps_update_speed > GPS_SPEED_THRESHOLD
)
gps_update_outlier[0] = False

# Full-length masks/arrays are kept for compatibility with the
# remainder of the script and for plotting against time_s.
gps_outlier = np.zeros(
    len(df),
    dtype=bool
)
gps_jump_distance = np.zeros(
    len(df),
    dtype=float
)
gps_jump_speed = np.zeros(
    len(df),
    dtype=float
)

gps_outlier[gps_update_indices] = gps_update_outlier
gps_jump_distance[gps_update_indices] = gps_update_distance
gps_jump_speed[gps_update_indices] = gps_update_speed


# ============================================================
# 5. Print GPS cleaning statistics
# ============================================================

print()
print("====================================")
print("STEP 10 — GPS CLEANING")
print("====================================")

print(
    f"GPS speed threshold: "
    f"{GPS_SPEED_THRESHOLD:.2f} m/s"
)

print(
    f"GPS coordinate updates: "
    f"{len(gps_update_indices) - 1}"
)

print(
    f"Detected GPS outliers: "
    f"{np.sum(gps_update_outlier)}"
)

print(
    f"Outlier percentage: "
    f"{100 * np.mean(gps_update_outlier):.3f}%"
)

print(
    f"Maximum GPS jump: "
    f"{np.max(gps_update_distance):.2f} m"
)

print(
    f"Maximum GPS jump speed: "
    f"{np.max(gps_update_speed):.2f} m/s"
)


# ============================================================
# 6. Create preliminary rejected GPS position
# ============================================================
# STEP 10 only rejects suspicious GPS updates. STEP 11 performs the
# actual interpolation and is the final cleaned GPS trajectory.

cleaned_north = np.zeros(len(df), dtype=float)
cleaned_east = np.zeros(len(df), dtype=float)

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

cleaned_north[0] = gps_north[0]
cleaned_east[0] = gps_east[0]

for i in range(1, len(df)):
    if gps_outlier[i]:
        cleaned_north[i] = cleaned_north[i - 1]
        cleaned_east[i] = cleaned_east[i - 1]
    else:
        cleaned_north[i] = gps_north[i]
        cleaned_east[i] = gps_east[i]

cleaned_lat = (
    lat0 + cleaned_north / lat_scale
)
cleaned_lon = (
    lon0 + cleaned_east / lon_scale
)


# ============================================================
# 7. Save STEP 10 diagnostic dataset
# ============================================================

cleaned_df = df.copy()
cleaned_df["cleaned_latitude"] = cleaned_lat
cleaned_df["cleaned_longitude"] = cleaned_lon
cleaned_df["gps_outlier"] = gps_outlier
cleaned_df["gps_jump_distance_m"] = gps_jump_distance
cleaned_df["gps_jump_speed_ms"] = gps_jump_speed

OUTPUT_PATH = "data/processed_S1_cleaned.csv"
cleaned_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 8. Plot original vs preliminary rejected GPS
# ============================================================

plt.figure(figsize=(10, 7))
plt.plot(
    gps_lon,
    gps_lat,
    label="Raw GPS"
)
plt.plot(
    cleaned_lon,
    cleaned_lat,
    label="Rejected GPS"
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Raw GPS vs Rejected GPS")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot GPS jump speed
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(
    time,
    gps_jump_speed,
    label="Raw GPS update speed"
)
plt.axhline(
    GPS_SPEED_THRESHOLD,
    linestyle="--",
    label="15 m/s threshold"
)
plt.xlabel("Time (seconds)")
plt.ylabel("GPS update speed (m/s)")
plt.title("GPS Update Speed and Outlier Threshold")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# STEP 10 COMPLETE
# ============================================================

print()
print("====================================")
print("STEP 10 COMPLETE")
print("====================================")
print("GPS outliers were identified using actual GPS update intervals.")
print("Rows were not deleted.")

# ============================================================
# STEP 11 — GPS OUTLIER REPLACEMENT
# ============================================================

print()
print("=" * 36)
print("STEP 11 — GPS OUTLIER REPLACEMENT")
print("=" * 36)

# Keep ORIGINAL GPS coordinates unchanged for comparison.
raw_gps_lat = df["latitude"].to_numpy(dtype=float).copy()
raw_gps_lon = df["longitude"].to_numpy(dtype=float).copy()

print(
    f"GPS outliers detected in STEP 10: "
    f"{int(np.sum(gps_update_outlier))}"
)

# ------------------------------------------------------------
# Work only on actual GPS update samples.
# ------------------------------------------------------------

corrected_update_lat = gps_update_lat.copy()
corrected_update_lon = gps_update_lon.copy()
correction_mask = np.zeros(
    len(gps_update_indices),
    dtype=bool
)

# Re-evaluate after each interpolation pass at the GPS-update level.
# This avoids the old bug where 10-Hz IMU rows were mistaken for
# independent GPS measurements.
for iteration in range(10):

    la1 = np.radians(corrected_update_lat[:-1])
    la2 = np.radians(corrected_update_lat[1:])
    dlon = np.radians(
        corrected_update_lon[1:]
        - corrected_update_lon[:-1]
    )
    dlat = la2 - la1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(la1)
        * np.cos(la2)
        * np.sin(dlon / 2.0) ** 2
    )

    distances = (
        2.0
        * 6371000.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1.0 - a)
        )
    )

    update_dt = np.diff(
        gps_update_time
    )

    update_speed = (
        distances
        / np.maximum(update_dt, 1e-6)
    )

    bad_updates = (
        update_speed > GPS_SPEED_THRESHOLD
    )

    if not np.any(bad_updates):
        break

    bad_indices = np.where(bad_updates)[0] + 1
    correction_mask[bad_indices] = True

    corrected_update_lat[bad_indices] = np.nan
    corrected_update_lon[bad_indices] = np.nan

    valid_lat = np.isfinite(corrected_update_lat)
    valid_lon = np.isfinite(corrected_update_lon)

    corrected_update_lat = np.interp(
        gps_update_time,
        gps_update_time[valid_lat],
        corrected_update_lat[valid_lat]
    )

    corrected_update_lon = np.interp(
        gps_update_time,
        gps_update_time[valid_lon],
        corrected_update_lon[valid_lon]
    )

# ------------------------------------------------------------
# Expand corrected GPS update values back over all sensor rows.
# GPS values are held constant between actual GPS updates.
# ------------------------------------------------------------

cleaned_lat = np.empty(len(df), dtype=float)
cleaned_lon = np.empty(len(df), dtype=float)

for i, start_index in enumerate(gps_update_indices):

    if i + 1 < len(gps_update_indices):
        end_index = gps_update_indices[i + 1]
    else:
        end_index = len(df)

    cleaned_lat[start_index:end_index] = corrected_update_lat[i]
    cleaned_lon[start_index:end_index] = corrected_update_lon[i]

# Put the final cleaned coordinates into the dataframe.
df["latitude"] = cleaned_lat
df["longitude"] = cleaned_lon
cleaned_time = time.copy()

# ------------------------------------------------------------
# Final GPS-update validation
# ------------------------------------------------------------

la1 = np.radians(corrected_update_lat[:-1])
la2 = np.radians(corrected_update_lat[1:])
dlon = np.radians(
    corrected_update_lon[1:]
    - corrected_update_lon[:-1]
)
dlat = la2 - la1

a = (
    np.sin(dlat / 2.0) ** 2
    + np.cos(la1)
    * np.cos(la2)
    * np.sin(dlon / 2.0) ** 2
)

final_update_distances = (
    2.0
    * 6371000.0
    * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a)
    )
)

final_update_dt = np.diff(
    gps_update_time
)

final_update_speed = (
    final_update_distances
    / np.maximum(final_update_dt, 1e-6)
)

remaining_update_outliers = (
    final_update_speed > GPS_SPEED_THRESHOLD
)

remaining_outlier_count = int(
    np.sum(remaining_update_outliers)
)

# Full-length diagnostic arrays.
cleaned_jumps = np.zeros(len(df), dtype=float)
cleaned_jump_speed = np.zeros(len(df), dtype=float)

cleaned_jumps[gps_update_indices[1:]] = final_update_distances
cleaned_jump_speed[gps_update_indices[1:]] = final_update_speed

cleaned_jump_speed[0] = 0.0

remaining_outliers = np.zeros(len(df), dtype=bool)
remaining_outliers[gps_update_indices[1:]] = remaining_update_outliers

# ------------------------------------------------------------
# Print replacement results
# ------------------------------------------------------------

print()
print("GPS REPLACEMENT RESULTS")
print("-" * 36)
print(f"Rows in dataset: {len(df)}")
print(
    f"GPS outliers originally detected: "
    f"{int(np.sum(gps_update_outlier))}"
)
print(
    f"GPS update coordinates corrected: "
    f"{int(np.sum(correction_mask))}"
)
print(
    f"Missing latitude values: "
    f"{int(df['latitude'].isna().sum())}"
)
print(
    f"Missing longitude values: "
    f"{int(df['longitude'].isna().sum())}"
)

print()
print("CLEANED GPS RESULTS")
print("-" * 36)
print(
    f"GPS update outliers remaining: "
    f"{remaining_outlier_count}"
)
print(
    f"Remaining outlier percentage: "
    f"{100.0 * remaining_outlier_count / max(1, len(gps_update_indices) - 1):.3f}%"
)
print(
    f"Maximum cleaned GPS jump: "
    f"{np.max(final_update_distances):.2f} m"
)
print(
    f"Maximum cleaned GPS jump speed: "
    f"{np.max(final_update_speed):.2f} m/s"
)

# Save the final cleaned dataset.
output_file = "data/processed_S1_cleaned.csv"

df["gps_outlier"] = remaining_outliers
df["gps_jump_distance_m"] = cleaned_jumps
df["gps_jump_speed_ms"] = cleaned_jump_speed

df.to_csv(
    output_file,
    index=False
)

print()
print(
    f"Cleaned dataset saved to: "
    f"{output_file}"
)

print()
print("=" * 36)
print("STEP 11 COMPLETE")
print("=" * 36)

if remaining_outlier_count == 0:
    print(
        "GPS outlier coordinates were replaced successfully."
    )
    print("No GPS speed outliers remain.")
else:
    print(
        "WARNING: Some GPS speed outliers still remain."
    )

print("Rows were not deleted.")


# ============================================================
# STEP 12 — GPS CLEANING VALIDATION
# ============================================================

print()
print("=" * 36)
print("STEP 12 — GPS CLEANING VALIDATION")
print("=" * 36)

print()
print("GPS VALIDATION RESULTS")
print("-" * 36)
print(f"Rows in dataset: {len(df)}")
print(
    f"GPS speed threshold: "
    f"{GPS_SPEED_THRESHOLD:.2f} m/s"
)
print(
    f"GPS update outliers remaining: "
    f"{remaining_outlier_count}"
)
print(
    f"Remaining outlier percentage: "
    f"{100.0 * remaining_outlier_count / max(1, len(gps_update_indices) - 1):.3f}%"
)
print(
    f"Maximum cleaned GPS jump: "
    f"{np.max(final_update_distances):.2f} m"
)
print(
    f"Maximum cleaned GPS jump speed: "
    f"{np.max(final_update_speed):.2f} m/s"
)

print()
print("VALIDATION CHECKS")
print("-" * 36)

if remaining_outlier_count == 0:
    print("PASS: No GPS speed outliers remain.")
else:
    print("FAIL: GPS speed outliers still remain.")

if np.all(np.isfinite(cleaned_lat)):
    print("PASS: All cleaned latitude values are valid.")
else:
    print("FAIL: Invalid cleaned latitude values detected.")

if np.all(np.isfinite(cleaned_lon)):
    print("PASS: All cleaned longitude values are valid.")
else:
    print("FAIL: Invalid cleaned longitude values detected.")

if np.max(final_update_speed) < GPS_SPEED_THRESHOLD:
    print(
        "PASS: Maximum cleaned GPS speed "
        "is below the threshold."
    )
else:
    print(
        "FAIL: Maximum cleaned GPS speed "
        "is still above the threshold."
    )

# ------------------------------------------------------------
# RAW VS CLEANED COMPARISON
# ------------------------------------------------------------

print()
print("RAW VS CLEANED COMPARISON")
print("-" * 36)

raw_max_jump = float(
    np.max(gps_update_distance)
)
raw_max_speed = float(
    np.max(gps_update_speed)
)
clean_max_jump = float(
    np.max(final_update_distances)
)
clean_max_speed = float(
    np.max(final_update_speed)
)

print(
    f"Maximum raw GPS jump: "
    f"{raw_max_jump:.2f} m"
)
print(
    f"Maximum cleaned GPS jump: "
    f"{clean_max_jump:.2f} m"
)
print(
    f"Maximum raw GPS jump speed: "
    f"{raw_max_speed:.2f} m/s"
)
print(
    f"Maximum cleaned GPS jump speed: "
    f"{clean_max_speed:.2f} m/s"
)

if clean_max_jump <= raw_max_jump:
    print(
        "PASS: Cleaned GPS maximum jump did not increase."
    )
else:
    print(
        "FAIL: Cleaned GPS maximum jump increased."
    )

if clean_max_speed <= raw_max_speed:
    print(
        "PASS: Cleaned GPS maximum jump speed did not increase."
    )
else:
    print(
        "FAIL: Cleaned GPS maximum jump speed increased."
    )

print()
print("=" * 36)
print("STEP 12 COMPLETE")
print("=" * 36)

if remaining_outlier_count == 0:
    print(
        "GPS cleaning validation completed successfully."
    )
else:
    print(
        "GPS cleaning validation detected "
        "remaining GPS speed outliers."
    )

# ============================================================
# STEP 13 — GPS TRAJECTORY SMOOTHING
# ============================================================

print("\n====================================")
print("STEP 13 — GPS TRAJECTORY SMOOTHING")
print("====================================")

# ------------------------------------------------------------
# Load the cleaned GPS coordinates
# ------------------------------------------------------------

smooth_lat = df["latitude"].to_numpy(dtype=float)
smooth_lon = df["longitude"].to_numpy(dtype=float)

# ------------------------------------------------------------
# Smoothing parameters
# ------------------------------------------------------------

SMOOTHING_WINDOW = 5

print(f"Smoothing window: {SMOOTHING_WINDOW} samples")

# ------------------------------------------------------------
# Apply rolling median smoothing
# ------------------------------------------------------------

lat_series = pd.Series(smooth_lat)
lon_series = pd.Series(smooth_lon)

smoothed_lat = (
    lat_series
    .rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
    .to_numpy()
)

smoothed_lon = (
    lon_series
    .rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
    .to_numpy()
)

# ------------------------------------------------------------
# Check for invalid values
# ------------------------------------------------------------

missing_smoothed_lat = np.sum(~np.isfinite(smoothed_lat))
missing_smoothed_lon = np.sum(~np.isfinite(smoothed_lon))

print("\nGPS SMOOTHING RESULTS")
print("------------------------------------")
print(f"Rows in dataset: {len(df)}")
print(f"Missing smoothed latitude values: {missing_smoothed_lat}")
print(f"Missing smoothed longitude values: {missing_smoothed_lon}")

# ------------------------------------------------------------
# Calculate movement before and after smoothing
# ------------------------------------------------------------

raw_smooth_jumps = np.zeros(len(df))
smoothed_jumps = np.zeros(len(df))

for i in range(1, len(df)):

    raw_smooth_jumps[i] = haversine_distance(
        smooth_lat[i - 1],
        smooth_lon[i - 1],
        smooth_lat[i],
        smooth_lon[i]
    )

    smoothed_jumps[i] = haversine_distance(
        smoothed_lat[i - 1],
        smoothed_lon[i - 1],
        smoothed_lat[i],
        smoothed_lon[i]
    )

max_before_smoothing = np.max(raw_smooth_jumps)
max_after_smoothing = np.max(smoothed_jumps)

print(f"Maximum GPS jump before smoothing: {max_before_smoothing:.2f} m")
print(f"Maximum GPS jump after smoothing: {max_after_smoothing:.2f} m")

# ------------------------------------------------------------
# Save smoothed coordinates into dataframe
# ------------------------------------------------------------

df["latitude_smoothed"] = smoothed_lat
df["longitude_smoothed"] = smoothed_lon

# ------------------------------------------------------------
# Save Step 13 dataset
# ------------------------------------------------------------

step13_output = "data/processed_S1_smoothed.csv"

df.to_csv(
    step13_output,
    index=False
)

print("\nVALIDATION CHECKS")
print("------------------------------------")

if missing_smoothed_lat == 0:
    print("PASS: All smoothed latitude values are valid.")
else:
    print("FAIL: Invalid smoothed latitude values detected.")

if missing_smoothed_lon == 0:
    print("PASS: All smoothed longitude values are valid.")
else:
    print("FAIL: Invalid smoothed longitude values detected.")

if max_after_smoothing <= max_before_smoothing:
    print("PASS: GPS maximum jump did not increase after smoothing.")
else:
    print("WARNING: GPS maximum jump increased after smoothing.")

# ------------------------------------------------------------
# Plot cleaned vs smoothed GPS
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    smooth_lon,
    smooth_lat,
    label="Cleaned GPS",
    linewidth=1
)

plt.plot(
    smoothed_lon,
    smoothed_lat,
    label="Smoothed GPS",
    linewidth=2
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Cleaned GPS vs Smoothed GPS")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 13 completion
# ------------------------------------------------------------

print("\n====================================")
print("STEP 13 COMPLETE")
print("====================================")
print("GPS trajectory smoothing completed.")
print("Rows were not deleted.")
print(f"Smoothed dataset saved to: {step13_output}")
# ============================================================
# STEP 14 — SMOOTHED GPS SPEED CALCULATION
# ============================================================

print()
print("====================================")
print("STEP 14 — SMOOTHED GPS SPEED CALCULATION")
print("====================================")

# GPS coordinates are updated less frequently than the IMU rows.
# Therefore calculate smoothed GPS speed between the actual GPS
# update timestamps, exactly as in STEP 10/11.

smoothed_update_lat = smoothed_lat[gps_update_indices]
smoothed_update_lon = smoothed_lon[gps_update_indices]

smoothed_update_distances = np.zeros(
    len(gps_update_indices),
    dtype=float
)

for i in range(1, len(gps_update_indices)):
    smoothed_update_distances[i] = haversine_distance(
        smoothed_update_lat[i - 1],
        smoothed_update_lon[i - 1],
        smoothed_update_lat[i],
        smoothed_update_lon[i]
    )

smoothed_update_dt = np.diff(
    gps_update_time,
    prepend=gps_update_time[0]
)
smoothed_update_dt[0] = 1.0
smoothed_update_dt = np.maximum(
    smoothed_update_dt,
    1e-6
)

smoothed_update_speed = (
    smoothed_update_distances
    / smoothed_update_dt
)
smoothed_update_speed[0] = 0.0

# Expand the update-level speed back to the full dataframe so the
# output column remains aligned with the original sensor rows.
smoothed_gps_jumps = np.zeros(len(df), dtype=float)
smoothed_gps_speed = np.zeros(len(df), dtype=float)

smoothed_gps_jumps[gps_update_indices] = (
    smoothed_update_distances
)
smoothed_gps_speed[gps_update_indices] = (
    smoothed_update_speed
)

# Store smoothed GPS speed in dataframe.
df["smoothed_gps_speed_mps"] = smoothed_gps_speed

print()
print("SMOOTHED GPS SPEED RESULTS")
print("-" * 36)
print(f"Rows in dataset: {len(df)}")
print(
    "Missing smoothed GPS speed values:",
    int(df["smoothed_gps_speed_mps"].isna().sum())
)
print(
    f"Maximum smoothed GPS jump: "
    f"{np.max(smoothed_update_distances):.2f} m"
)
print(
    f"Maximum smoothed GPS speed: "
    f"{np.max(smoothed_update_speed):.2f} m/s"
)
print(
    f"Average smoothed GPS speed: "
    f"{np.mean(smoothed_update_speed):.2f} m/s"
)

max_speed_kmh = np.max(smoothed_update_speed) * 3.6
average_speed_kmh = np.mean(smoothed_update_speed) * 3.6

print(
    f"Maximum smoothed GPS speed: "
    f"{max_speed_kmh:.2f} km/h"
)
print(
    f"Average smoothed GPS speed: "
    f"{average_speed_kmh:.2f} km/h"
)

print()
print("VALIDATION CHECKS")
print("-" * 36)

if df["smoothed_gps_speed_mps"].isna().sum() == 0:
    print("PASS: No missing smoothed GPS speed values.")
else:
    print("FAIL: Missing smoothed GPS speed values detected.")

if np.min(smoothed_gps_speed) >= 0:
    print("PASS: No negative GPS speed values.")
else:
    print("FAIL: Negative GPS speed values detected.")

if np.isfinite(smoothed_gps_speed).all():
    print("PASS: All smoothed GPS speeds are finite.")
else:
    print("FAIL: Invalid GPS speed values detected.")

print()
print("RAW VS SMOOTHED GPS SPEED")
print("-" * 36)

raw_max_speed = float(
    np.max(gps_update_speed)
)
smoothed_max_speed = float(
    np.max(smoothed_update_speed)
)

print(
    f"Maximum raw GPS speed: "
    f"{raw_max_speed:.2f} m/s"
)
print(
    f"Maximum smoothed GPS speed: "
    f"{smoothed_max_speed:.2f} m/s"
)

if smoothed_max_speed <= raw_max_speed:
    print(
        "PASS: Smoothed GPS maximum speed did not increase."
    )
else:
    print(
        "WARNING: Smoothed GPS maximum speed increased."
    )

smoothed_output_path = "data/processed_S1_smoothed.csv"

df.to_csv(
    smoothed_output_path,
    index=False
)

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    smoothed_gps_speed,
    label="Smoothed GPS speed"
)

plt.axhline(
    y=GPS_SPEED_THRESHOLD,
    linestyle="--",
    label="15 m/s threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Smoothed GPS speed (m/s)")
plt.title("Smoothed GPS Speed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print()
print("====================================")
print("STEP 14 COMPLETE")
print("====================================")
print("Smoothed GPS speed calculation completed.")
print(
    f"Smoothed dataset saved to: "
    f"{smoothed_output_path}"
)

print()
print("Run this script and send me the COMPLETE terminal output.")