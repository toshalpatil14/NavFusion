import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = (
    ROOT / "data" / "raw" / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset" / "S-Vw4.csv"
)

DR_FILE = (
    ROOT / "results"
    / "dead_reckoning_trajectory.csv"
)

RESULTS = ROOT / "results"

METRICS_FILE = (
    RESULTS / "filtered_gps_dr_metrics.txt"
)

PLOT_FILE = (
    RESULTS / "filtered_gps_vs_dr.png"
)

COMPARISON_FILE = (
    RESULTS / "filtered_gps_dr_comparison.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# GPS coordinate movement is accepted only if it is not
# grossly inconsistent with the recorded GPS speed.
#
# We allow a generous factor because GPS coordinates themselves
# contain noise.
SPEED_FACTOR = 3.0

# Minimum speed floor to avoid division problems at STOP.
MIN_SPEED_KMH = 1.0

# Absolute maximum physically plausible GPS speed.
MAX_SPEED_KMH = 100.0


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("FILTERED GPS vs DEAD RECKONING")
print("=" * 60)

gps = pd.read_csv(
    GPS_FILE,
    encoding="latin1"
)

gps.columns = gps.columns.str.strip()

dr = pd.read_csv(DR_FILE)

print("\nGPS rows:", len(gps))
print("DR rows:", len(dr))


# ============================================================
# GPS COLUMNS
# ============================================================

time_col = "TIME SINCE START (ms)"
lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"
speed_col = "GPS SPEED (Kmh)"


gps = gps[
    [
        time_col,
        lat_col,
        lon_col,
        speed_col,
    ]
].copy()

gps = gps.apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

gps = gps.rename(
    columns={
        time_col: "timestamp_ms"
    }
)

gps = gps.reset_index(drop=True)


# ============================================================
# GPS CONSECUTIVE MOVEMENT
# ============================================================

R = 6371000.0

lat = np.deg2rad(
    gps[lat_col].to_numpy()
)

lon = np.deg2rad(
    gps[lon_col].to_numpy()
)

timestamp = (
    gps["timestamp_ms"]
    .to_numpy()
)

gps_speed = (
    gps[speed_col]
    .to_numpy()
)

dt = np.diff(timestamp) / 1000.0

dx = (
    np.diff(lon)
    * R
    * np.cos(lat[:-1])
)

dy = (
    np.diff(lat)
    * R
)

gps_step = np.sqrt(
    dx ** 2 +
    dy ** 2
)

# Speed implied by coordinates.
coordinate_speed = np.zeros(
    len(gps)
)

valid_dt = dt > 0

coordinate_speed[1:][valid_dt] = (
    gps_step[valid_dt]
    / dt[valid_dt]
    * 3.6
)


# ============================================================
# PHYSICAL GPS FILTER
# ============================================================

# Expected distance during each interval from recorded GPS speed.
expected_step = (
    gps_speed[:-1]
    / 3.6
    * dt
)

# Allow a generous multiplier.
allowed_step = np.maximum(
    expected_step * SPEED_FACTOR,
    MIN_SPEED_KMH / 3.6 * dt
)

# A movement is valid when:
#
# 1. timestamp interval is valid
# 2. coordinate displacement is not impossible
# 3. coordinate-derived speed is below absolute limit

valid_step = (
    valid_dt
    & (
        gps_step <= allowed_step
    )
    & (
        coordinate_speed[1:]
        <= MAX_SPEED_KMH
    )
)

valid_step = np.insert(
    valid_step,
    0,
    True
)

print(
    "\nValid GPS coordinate samples:",
    int(valid_step.sum())
)

print(
    "Rejected GPS coordinate samples:",
    int((~valid_step).sum())
)


# ============================================================
# INTERPOLATE INVALID GPS POSITIONS
# ============================================================

gps.loc[
    ~valid_step,
    [lat_col, lon_col]
] = np.nan

gps[
    [lat_col, lon_col]
] = (
    gps[
        [lat_col, lon_col]
    ]
    .interpolate()
    .bfill()
    .ffill()
)


# ============================================================
# MATCH DR TIMESTAMPS
# ============================================================

merged = pd.merge(
    dr,
    gps[
        [
            "timestamp_ms",
            lat_col,
            lon_col,
            speed_col,
        ]
    ],
    on="timestamp_ms",
    how="inner"
)

print(
    "DR/GPS matched samples:",
    len(merged)
)


# ============================================================
# GPS → LOCAL XY
# ============================================================

gps_lat = np.deg2rad(
    merged[lat_col].to_numpy()
)

gps_lon = np.deg2rad(
    merged[lon_col].to_numpy()
)

lat0 = gps_lat[0]
lon0 = gps_lon[0]

gps_x = (
    (gps_lon - lon0)
    * R
    * np.cos(lat0)
)

gps_y = (
    (gps_lat - lat0)
    * R
)


# ============================================================
# DR XY
# ============================================================

dr_x = merged[
    "x_m"
].to_numpy(copy=True)

dr_y = merged[
    "y_m"
].to_numpy(copy=True)

dr_x -= dr_x[0]
dr_y -= dr_y[0]


# ============================================================
# ALIGN GPS
# ============================================================

gps_x -= gps_x[0]
gps_y -= gps_y[0]


# ============================================================
# POSITION ERROR
# ============================================================

error_x = dr_x - gps_x
error_y = dr_y - gps_y

position_error = np.sqrt(
    error_x ** 2 +
    error_y ** 2
)


# ============================================================
# DISTANCES
# ============================================================

gps_step_filtered = np.sqrt(
    np.diff(gps_x) ** 2 +
    np.diff(gps_y) ** 2
)

dr_step = np.sqrt(
    np.diff(dr_x) ** 2 +
    np.diff(dr_y) ** 2
)

gps_distance = np.sum(
    gps_step_filtered
)

dr_distance = np.sum(
    dr_step
)


# ============================================================
# METRICS
# ============================================================

mean_error = np.mean(
    position_error
)

median_error = np.median(
    position_error
)

rmse = np.sqrt(
    np.mean(
        position_error ** 2
    )
)

p90 = np.percentile(
    position_error,
    90
)

max_error = np.max(
    position_error
)

final_error = position_error[-1]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FILTERED GPS vs DR RESULTS")
print("=" * 60)

print(
    f"Matched samples       : {len(merged)}"
)

print(
    f"GPS samples rejected  : "
    f"{int((~valid_step).sum())}"
)

print(
    f"Filtered GPS distance : "
    f"{gps_distance:.2f} m"
)

print(
    f"DR distance           : "
    f"{dr_distance:.2f} m"
)

print(
    f"Mean position error   : "
    f"{mean_error:.2f} m"
)

print(
    f"Median position error : "
    f"{median_error:.2f} m"
)

print(
    f"RMSE position error   : "
    f"{rmse:.2f} m"
)

print(
    f"90th percentile error : "
    f"{p90:.2f} m"
)

print(
    f"Maximum position error: "
    f"{max_error:.2f} m"
)

print(
    f"Final position error  : "
    f"{final_error:.2f} m"
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FILTERED GPS vs DEAD RECKONING\n"
    )

    f.write(
        "===============================\n"
    )

    f.write(
        f"Matched samples: {len(merged)}\n"
    )

    f.write(
        f"GPS samples rejected: "
        f"{int((~valid_step).sum())}\n"
    )

    f.write(
        f"Filtered GPS distance: "
        f"{gps_distance:.4f} m\n"
    )

    f.write(
        f"DR distance: "
        f"{dr_distance:.4f} m\n"
    )

    f.write(
        f"Mean position error: "
        f"{mean_error:.4f} m\n"
    )

    f.write(
        f"Median position error: "
        f"{median_error:.4f} m\n"
    )

    f.write(
        f"RMSE position error: "
        f"{rmse:.4f} m\n"
    )

    f.write(
        f"90th percentile error: "
        f"{p90:.4f} m\n"
    )

    f.write(
        f"Maximum position error: "
        f"{max_error:.4f} m\n"
    )

    f.write(
        f"Final position error: "
        f"{final_error:.4f} m\n"
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "timestamp_ms":
        merged["timestamp_ms"],

    "dr_x_m":
        dr_x,

    "dr_y_m":
        dr_y,

    "gps_x_m":
        gps_x,

    "gps_y_m":
        gps_y,

    "position_error_m":
        position_error,

})

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(
    gps_x,
    gps_y,
    label="Filtered GPS"
)

plt.plot(
    dr_x,
    dr_y,
    label="Dead Reckoning"
)

plt.scatter(
    gps_x[0],
    gps_y[0],
    marker="o",
    label="Start"
)

plt.scatter(
    gps_x[-1],
    gps_y[-1],
    marker="x",
    label="End"
)

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "Filtered GPS vs Dead Reckoning - S-Vw4"
)

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nFiles saved:")
print(METRICS_FILE)
print(PLOT_FILE)
print(COMPARISON_FILE)

print("\n" + "=" * 60)
print("FILTERED GPS EVALUATION COMPLETE")
print("=" * 60)