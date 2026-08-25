import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IO-VNBD"
    / "Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
    / "S-Dataset"
    / "S-Vw4.csv"
)

DR_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_trajectory.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_FILE = (
    RESULTS_DIR
    / "dead_reckoning_metrics.txt"
)

PLOT_FILE = (
    RESULTS_DIR
    / "dead_reckoning_vs_gps.png"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "dead_reckoning_comparison.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Maximum allowed GPS displacement between consecutive
# raw samples before considering it an outlier.
MAX_GPS_JUMP_M = 100.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DEAD RECKONING vs CLEAN GPS EVALUATION")
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

TIME_COLUMN = "TIME SINCE START (ms)"
LAT_COLUMN = "GPS LATITUDE (degrees)"

longitude_candidates = [
    c for c in gps.columns
    if "GPS LONGITUDE" in c
]

if not longitude_candidates:
    raise ValueError(
        "GPS longitude column not found."
    )

LON_COLUMN = longitude_candidates[0]


# ============================================================
# CLEAN GPS DATA
# ============================================================

gps = gps[
    [
        TIME_COLUMN,
        LAT_COLUMN,
        LON_COLUMN,
    ]
].copy()

gps[TIME_COLUMN] = pd.to_numeric(
    gps[TIME_COLUMN],
    errors="coerce"
)

gps[LAT_COLUMN] = pd.to_numeric(
    gps[LAT_COLUMN],
    errors="coerce"
)

gps[LON_COLUMN] = pd.to_numeric(
    gps[LON_COLUMN],
    errors="coerce"
)

gps = gps.dropna().reset_index(drop=True)

gps = gps.rename(
    columns={
        TIME_COLUMN: "timestamp_ms"
    }
)

print(
    "GPS rows after numeric cleaning:",
    len(gps)
)


# ============================================================
# GPS CONSECUTIVE DISPLACEMENT
# ============================================================

lat = gps[LAT_COLUMN].to_numpy()
lon = gps[LON_COLUMN].to_numpy()

lat_prev = lat[:-1]
lon_prev = lon[:-1]

lat_rad = np.deg2rad(lat_prev)

dlat = np.deg2rad(
    np.diff(lat)
)

dlon = np.deg2rad(
    np.diff(lon)
)

EARTH_RADIUS = 6371000.0

dx = (
    dlon
    * EARTH_RADIUS
    * np.cos(lat_rad)
)

dy = (
    dlat
    * EARTH_RADIUS
)

gps_jump = np.sqrt(
    dx ** 2 +
    dy ** 2
)

gps_jump = np.insert(
    gps_jump,
    0,
    0.0
)

gps["gps_jump_m"] = gps_jump


# ============================================================
# REMOVE OBVIOUS GPS JUMPS
# ============================================================

bad_jump = (
    gps["gps_jump_m"] >
    MAX_GPS_JUMP_M
)

bad_count = int(
    bad_jump.sum()
)

print(
    f"\nGPS jumps > {MAX_GPS_JUMP_M:.0f} m:",
    bad_count
)


gps.loc[
    bad_jump,
    [LAT_COLUMN, LON_COLUMN]
] = np.nan


# Interpolate removed coordinates over time.
gps[
    [LAT_COLUMN, LON_COLUMN]
] = (
    gps[
        [LAT_COLUMN, LON_COLUMN]
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
            LAT_COLUMN,
            LON_COLUMN,
        ]
    ],
    on="timestamp_ms",
    how="inner"
)

print(
    "Exact timestamp matches:",
    len(merged)
)


if len(merged) == 0:

    raise RuntimeError(
        "No timestamps matched."
    )


# ============================================================
# GPS → LOCAL METRES
# ============================================================

lat0 = merged[LAT_COLUMN].iloc[0]
lon0 = merged[LON_COLUMN].iloc[0]

lat0_rad = np.deg2rad(lat0)

gps_lat_rad = np.deg2rad(
    merged[LAT_COLUMN].to_numpy()
)

gps_lon_rad = np.deg2rad(
    merged[LON_COLUMN].to_numpy()
)

gps_x = (
    (
        gps_lon_rad -
        np.deg2rad(lon0)
    )
    * EARTH_RADIUS
    * np.cos(lat0_rad)
)

gps_y = (
    (
        gps_lat_rad -
        np.deg2rad(lat0)
    )
    * EARTH_RADIUS
)


# ============================================================
# DR TRAJECTORY
# ============================================================

dr_x = merged["x_m"].to_numpy(copy=True)
dr_y = merged["y_m"].to_numpy(copy=True)


# Align both trajectories at origin.

dr_x -= dr_x[0]
dr_y -= dr_y[0]

gps_x -= gps_x[0]
gps_y -= gps_y[0]


# ============================================================
# POSITION ERROR
# ============================================================

error_x = (
    dr_x -
    gps_x
)

error_y = (
    dr_y -
    gps_y
)

position_error = np.sqrt(
    error_x ** 2 +
    error_y ** 2
)


# ============================================================
# TRAJECTORY DISTANCES
# ============================================================

gps_step = np.sqrt(
    np.diff(gps_x) ** 2 +
    np.diff(gps_y) ** 2
)

dr_step = np.sqrt(
    np.diff(dr_x) ** 2 +
    np.diff(dr_y) ** 2
)

gps_distance = np.sum(
    gps_step
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

p90_error = np.percentile(
    position_error,
    90
)

max_error = np.max(
    position_error
)

final_error = (
    position_error[-1]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLEAN GPS vs DEAD RECKONING")
print("=" * 60)

print(
    f"Matched samples       : {len(merged)}"
)

print(
    f"GPS jumps removed     : {bad_count}"
)

print(
    f"GPS trajectory length : "
    f"{gps_distance:.2f} m"
)

print(
    f"DR trajectory length  : "
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
    f"{p90_error:.2f} m"
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
        "CLEAN GPS vs DEAD RECKONING\n"
    )

    f.write(
        "===========================\n"
    )

    f.write(
        f"Matched samples: "
        f"{len(merged)}\n"
    )

    f.write(
        f"GPS jumps removed: "
        f"{bad_count}\n"
    )

    f.write(
        f"GPS trajectory length: "
        f"{gps_distance:.4f} m\n"
    )

    f.write(
        f"DR trajectory length: "
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
        f"{p90_error:.4f} m\n"
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
# SAVE COMPARISON DATA
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
    label="Clean GPS"
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
    "Dead Reckoning vs Clean GPS - S-Vw4"
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
print("CLEAN GPS EVALUATION COMPLETE")
print("=" * 60)