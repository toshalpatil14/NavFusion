import pandas as pd
import os

# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

file_path = (
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "S (Driver A)/S1/S-S1.csv"
)

output_path = "data/processed_S1.csv"

print("Loading dataset...")

# The dataset uses a non-UTF8 encoding
df = pd.read_csv(
    file_path,
    encoding="latin1"
)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

print("Dataset loaded!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# RENAME IMPORTANT COLUMNS
# --------------------------------------------------

rename_map = {

    # GPS
    "GPS LATITUDE (degrees)": "latitude",
    "GPS LONGITUDE (degrees)": "longitude",
    "GPS ALTITUDE (m)": "altitude",
    "GPS SPEED (Kmh)": "speed_kmh",
    "GPS ACCURACY (m)": "gps_accuracy",
    "GPS ORIENTATION (Â°)": "gps_orientation",
    "GPS SATELLITES IN RANGE": "satellites",

    # Time
    "TIME SINCE START (ms)": "time_ms",

    # Accelerometer
    "ACCELEROMETER X (m/s²)": "accel_x",
    "ACCELEROMETER Y (m/s²)": "accel_y",
    "ACCELEROMETER Z (m/s²)": "accel_z",

    # Gravity
    "GRAVITY X (m/s²)": "gravity_x",
    "GRAVITY Y (m/s²)": "gravity_y",
    "GRAVITY Z (m/s²)": "gravity_z",

    # Gyroscope
    "GYROSCOPE Yaw (rad/s)": "gyro_yaw",
    "GYROSCOPE Pitch (rad/s)": "gyro_pitch",
    "GYROSCOPE Roll (rad/s)": "gyro_roll",

    # Magnetic field
    "MAGNETIC FIELD X (µT)": "mag_x",
    "MAGNETIC FIELD Y (µT)": "mag_y",
    "MAGNETIC FIELD Z (µT)": "mag_z",

    # Orientation
    "ORIENTATION (Yaw) (Â°)": "orientation_yaw",
    "ORIENTATION (Pitch) (Â°)": "orientation_pitch",
    "ORIENTATION (Roll ) (Â°)": "orientation_roll",
}


# Only rename columns that actually exist
df.rename(
    columns=rename_map,
    inplace=True
)

print()
print("Columns after renaming:")
print(df.columns.tolist())


# --------------------------------------------------
# CONVERT TIME
# --------------------------------------------------

df["time_s"] = (
    df["time_ms"] / 1000.0
)


# --------------------------------------------------
# CONVERT GPS SPEED
# --------------------------------------------------

# Original GPS speed is in km/h.
# Convert it to metres/second.

df["gps_speed_ms"] = (
    df["speed_kmh"] / 3.6
)
# Backward-compatible name used by older experiments
df["speed_ms"] = df["gps_speed_ms"]

# --------------------------------------------------
# DATA QUALITY CHECK
# --------------------------------------------------

print()
print("==============================")
print("DATA QUALITY CHECK")
print("==============================")

print(
    "Missing values:",
    df.isna().sum().sum()
)

print(
    "Time start:",
    df["time_s"].iloc[0],
    "seconds"
)

print(
    "Time end:",
    df["time_s"].iloc[-1],
    "seconds"
)

print(
    "GPS speed mean:",
    f"{df['gps_speed_ms'].mean():.2f} m/s"
)

print(
    "GPS speed maximum:",
    f"{df['gps_speed_ms'].max():.2f} m/s"
)


# --------------------------------------------------
# SAVE PROCESSED DATASET
# --------------------------------------------------

df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("==============================")
print("PROCESSED DATASET")
print("==============================")

print(
    "Saved to:",
    output_path
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print()
print("Important columns:")

important_columns = [

    "latitude",
    "longitude",

    "altitude",

    "speed_kmh",
    "gps_speed_ms",

    "gps_accuracy",
    "gps_orientation",
    "satellites",

    "time_ms",
    "time_s",

    "accel_x",
    "accel_y",
    "accel_z",

    "gravity_x",
    "gravity_y",
    "gravity_z",

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    "mag_x",
    "mag_y",
    "mag_z",

    "orientation_yaw",
    "orientation_pitch",
    "orientation_roll",
]


for column in important_columns:

    if column in df.columns:
        print("✓", column)


print()
print("Preparation complete!")
