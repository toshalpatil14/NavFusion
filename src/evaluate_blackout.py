import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NAV_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gps_reference_10hz.csv"
)

DR_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_v2_trajectory.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "blackout_evaluation.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "blackout_metrics.txt"
)


# ============================================================
# BLACKOUT WINDOW
# ============================================================

BLACKOUT_START = 5067310
BLACKOUT_END = 6332010


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("GNSS BLACKOUT EVALUATION")
print("=" * 60)

nav = pd.read_csv(NAV_FILE)
dr = pd.read_csv(DR_FILE)

print(f"\nNavigation rows: {len(nav)}")
print(f"DR rows        : {len(dr)}")


# ============================================================
# SELECT BLACKOUT
# ============================================================

nav = nav[
    (nav["timestamp_ms"] >= BLACKOUT_START)
    & (nav["timestamp_ms"] <= BLACKOUT_END)
].copy()

dr = dr[
    (dr["timestamp_ms"] >= BLACKOUT_START)
    & (dr["timestamp_ms"] <= BLACKOUT_END)
].copy()


# ============================================================
# MERGE
# ============================================================

merged = pd.merge(
    nav[
        [
            "timestamp_ms",
            "gps_latitude",
            "gps_longitude",
            "gps_speed_kmh",
        ]
    ],
    dr[
        [
            "timestamp_ms",
            "x_m",
            "y_m",
        ]
    ],
    on="timestamp_ms",
    how="inner",
)

blackout_total = len(merged)

blackout_missing_gps = merged[
    ["gps_latitude", "gps_longitude"]
].isna().any(axis=1).sum()

print(
    f"\nBlackout samples without GPS: "
    f"{blackout_missing_gps}"
)

merged = merged.dropna(
    subset=[
        "gps_latitude",
        "gps_longitude",
    ]
).reset_index(drop=True)

print(
    f"Matched valid blackout samples: "
    f"{len(merged)}"
)
# ============================================================
# GPS → LOCAL METRES
# ============================================================

EARTH_RADIUS_M = 6371000.0

lat = np.deg2rad(
    merged["gps_latitude"].to_numpy()
)

lon = np.deg2rad(
    merged["gps_longitude"].to_numpy()
)

lat0 = lat[0]
lon0 = lon[0]

gps_x = (
    (lon - lon0)
    * EARTH_RADIUS_M
    * np.cos(lat0)
)

gps_y = (
    (lat - lat0)
    * EARTH_RADIUS_M
)


# ============================================================
# DR → LOCAL ORIGIN
# ============================================================

dr_x = (
    merged["x_m"].to_numpy(copy=True)
)

dr_y = (
    merged["y_m"].to_numpy(copy=True)
)

dr_x -= dr_x[0]
dr_y -= dr_y[0]


# ============================================================
# POSITION ERROR
# ============================================================

error = np.sqrt(
    (dr_x - gps_x) ** 2
    +
    (dr_y - gps_y) ** 2
)

merged["gps_x_m"] = gps_x
merged["gps_y_m"] = gps_y

merged["dr_x_relative_m"] = dr_x
merged["dr_y_relative_m"] = dr_y

merged["position_error_m"] = error


# ============================================================
# METRICS
# ============================================================

mae = np.mean(error)

median = np.median(error)

rmse = np.sqrt(
    np.mean(error ** 2)
)

p90 = np.percentile(
    error,
    90
)

maximum = np.max(error)

final_error = error[-1]

gps_distance = np.sum(
    np.sqrt(
        np.diff(gps_x) ** 2
        +
        np.diff(gps_y) ** 2
    )
)

dr_distance = np.sum(
    np.sqrt(
        np.diff(dr_x) ** 2
        +
        np.diff(dr_y) ** 2
    )
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 60)
print("BLACKOUT RESULTS")
print("=" * 60)

print(
    f"Blackout start       : "
    f"{BLACKOUT_START} ms"
)

print(
    f"Blackout end         : "
    f"{BLACKOUT_END} ms"
)

print(
    f"Matched samples      : "
    f"{len(merged)}"
)

print(
    f"GPS distance         : "
    f"{gps_distance:.2f} m"
)

print(
    f"DR distance          : "
    f"{dr_distance:.2f} m"
)

print(
    f"Mean position error  : "
    f"{mae:.2f} m"
)

print(
    f"Median error         : "
    f"{median:.2f} m"
)

print(
    f"RMSE                 : "
    f"{rmse:.2f} m"
)

print(
    f"90th percentile      : "
    f"{p90:.2f} m"
)

print(
    f"Maximum error        : "
    f"{maximum:.2f} m"
)

print(
    f"Final position error : "
    f"{final_error:.2f} m"
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

metrics = f"""
GNSS BLACKOUT EVALUATION

Blackout start: {BLACKOUT_START} ms
Blackout end: {BLACKOUT_END} ms

Matched samples: {len(merged)}

GPS distance: {gps_distance:.2f} m
DR distance: {dr_distance:.2f} m

Mean position error: {mae:.2f} m
Median position error: {median:.2f} m
RMSE position error: {rmse:.2f} m
90th percentile error: {p90:.2f} m
Maximum position error: {maximum:.2f} m
Final position error: {final_error:.2f} m
"""

METRICS_FILE.write_text(
    metrics.strip() + "\n",
    encoding="utf-8"
)


print("\nFiles saved:")
print(OUTPUT_FILE)
print(METRICS_FILE)

print("\n" + "=" * 60)
print("BLACKOUT EVALUATION COMPLETE")
print("=" * 60)