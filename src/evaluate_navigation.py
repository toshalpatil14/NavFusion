import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GPS_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gps_reference_10hz.csv"
)

DR_FILE = (
    PROJECT_ROOT
    / "results"
    / "dead_reckoning_v2_trajectory.csv"
)

FUSION_FILE = (
    PROJECT_ROOT
    / "results"
    / "fusion_v1_trajectory.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_evaluation.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_metrics.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

BLACKOUT_START = 5067310
BLACKOUT_END = 6332010

EARTH_RADIUS_M = 6371000.0


# ============================================================
# HELPERS
# ============================================================

def gps_to_local_xy(lat, lon):
    """
    Convert GPS latitude/longitude to local East/North metres.

    The first valid GPS sample becomes the local origin.
    """

    lat = np.deg2rad(
        np.asarray(lat, dtype=float)
    )

    lon = np.deg2rad(
        np.asarray(lon, dtype=float)
    )

    lat0 = lat[0]
    lon0 = lon[0]

    x = (
        (lon - lon0)
        * EARTH_RADIUS_M
        * np.cos(lat0)
    )

    y = (
        (lat - lat0)
        * EARTH_RADIUS_M
    )

    return x, y


def calculate_metrics(
    reference_x,
    reference_y,
    estimate_x,
    estimate_y,
):
    """
    Calculate position-error metrics.
    """

    error = np.sqrt(
        (estimate_x - reference_x) ** 2
        +
        (estimate_y - reference_y) ** 2
    )

    return {
        "samples": len(error),
        "mean_error_m": float(
            np.mean(error)
        ),
        "median_error_m": float(
            np.median(error)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "p90_error_m": float(
            np.percentile(error, 90)
        ),
        "max_error_m": float(
            np.max(error)
        ),
        "final_error_m": float(
            error[-1]
        ),
    }


def trajectory_distance(x, y):
    """
    Calculate total trajectory distance.
    """

    dx = np.diff(x)
    dy = np.diff(y)

    return float(
        np.sum(
            np.sqrt(
                dx ** 2 +
                dy ** 2
            )
        )
    )


def print_metrics(name, metrics):
    """
    Print a metrics dictionary.
    """

    print(f"\n{name}:")

    for key, value in metrics.items():

        if key == "samples":
            print(
                f"{key:20s}: {value}"
            )

        else:
            print(
                f"{key:20s}: {value:.2f}"
            )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("NAVIGATION EVALUATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    GPS_FILE,
    DR_FILE,
    FUSION_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

gps = pd.read_csv(
    GPS_FILE
)

dr = pd.read_csv(
    DR_FILE
)

fusion = pd.read_csv(
    FUSION_FILE
)


print("\nInput datasets:")

print(
    f"GPS reference : {len(gps)}"
)

print(
    f"DR V2         : {len(dr)}"
)

print(
    f"Fusion V1     : {len(fusion)}"
)


# ============================================================
# VALIDATE GPS
# ============================================================

gps_required = [
    "timestamp_ms",
    "gps_latitude",
    "gps_longitude",
]

for column in gps_required:

    if column not in gps.columns:

        raise ValueError(
            f"Missing GPS column: {column}"
        )


# ============================================================
# VALIDATE DR
# ============================================================

dr_required = [
    "timestamp_ms",
    "x_m",
    "y_m",
]

for column in dr_required:

    if column not in dr.columns:

        raise ValueError(
            f"Missing DR column: {column}"
        )


# ============================================================
# VALIDATE FUSION
# ============================================================

fusion_required = [
    "timestamp_ms",
    "x_m",
    "y_m",
]

for column in fusion_required:

    if column not in fusion.columns:

        raise ValueError(
            f"Missing Fusion column: {column}"
        )


# ============================================================
# CLEAN GPS
# ============================================================

gps = gps[
    gps_required
].copy()

gps = gps.dropna(
    subset=[
        "gps_latitude",
        "gps_longitude",
    ]
)


# ============================================================
# RENAME TRAJECTORY COLUMNS
# ============================================================

dr = dr[
    dr_required
].copy()

dr = dr.rename(
    columns={
        "x_m": "dr_x_m",
        "y_m": "dr_y_m",
    }
)


fusion = fusion[
    fusion_required
].copy()

fusion = fusion.rename(
    columns={
        "x_m": "fusion_x_m",
        "y_m": "fusion_y_m",
    }
)


# ============================================================
# REMOVE DUPLICATE TIMESTAMPS
# ============================================================

gps = gps.drop_duplicates(
    subset=["timestamp_ms"]
)

dr = dr.drop_duplicates(
    subset=["timestamp_ms"]
)

fusion = fusion.drop_duplicates(
    subset=["timestamp_ms"]
)


# ============================================================
# MERGE
# ============================================================

merged = (
    gps
    .merge(
        dr,
        on="timestamp_ms",
        how="inner",
    )
    .merge(
        fusion,
        on="timestamp_ms",
        how="inner",
    )
)


print(
    f"\nMatched samples: {len(merged)}"
)


if len(merged) == 0:

    raise RuntimeError(
        "No matching GPS/DR/Fusion timestamps."
    )


# ============================================================
# SORT BY TIME
# ============================================================

merged = merged.sort_values(
    "timestamp_ms"
).reset_index(
    drop=True
)


# ============================================================
# GPS → LOCAL METRES
# ============================================================

gps_x, gps_y = gps_to_local_xy(
    merged["gps_latitude"].to_numpy(),
    merged["gps_longitude"].to_numpy(),
)


# ============================================================
# DR TRAJECTORY
# ============================================================

dr_x = (
    merged["dr_x_m"]
    .to_numpy(copy=True)
)

dr_y = (
    merged["dr_y_m"]
    .to_numpy(copy=True)
)


# ============================================================
# FUSION TRAJECTORY
# ============================================================

fusion_x = (
    merged["fusion_x_m"]
    .to_numpy(copy=True)
)

fusion_y = (
    merged["fusion_y_m"]
    .to_numpy(copy=True)
)


# ============================================================
# GLOBAL ORIGIN ALIGNMENT
# ============================================================

dr_x -= dr_x[0]
dr_y -= dr_y[0]

fusion_x -= fusion_x[0]
fusion_y -= fusion_y[0]


# ============================================================
# OVERALL METRICS
# ============================================================

dr_metrics = calculate_metrics(
    gps_x,
    gps_y,
    dr_x,
    dr_y,
)

fusion_metrics = calculate_metrics(
    gps_x,
    gps_y,
    fusion_x,
    fusion_y,
)


# ============================================================
# OVERALL DISTANCES
# ============================================================

gps_distance = trajectory_distance(
    gps_x,
    gps_y,
)

dr_distance = trajectory_distance(
    dr_x,
    dr_y,
)

fusion_distance = trajectory_distance(
    fusion_x,
    fusion_y,
)


# ============================================================
# BLACKOUT SELECTION
# ============================================================

blackout_mask = (
    (merged["timestamp_ms"] >= BLACKOUT_START)
    &
    (merged["timestamp_ms"] <= BLACKOUT_END)
).to_numpy()


blackout_count = int(
    blackout_mask.sum()
)


if blackout_count == 0:

    raise RuntimeError(
        "No blackout samples found."
    )


# ============================================================
# BLACKOUT TRAJECTORIES
# ============================================================

gps_x_blackout = (
    gps_x[blackout_mask]
    .copy()
)

gps_y_blackout = (
    gps_y[blackout_mask]
    .copy()
)

dr_x_blackout = (
    dr_x[blackout_mask]
    .copy()
)

dr_y_blackout = (
    dr_y[blackout_mask]
    .copy()
)

fusion_x_blackout = (
    fusion_x[blackout_mask]
    .copy()
)

fusion_y_blackout = (
    fusion_y[blackout_mask]
    .copy()
)


# ============================================================
# BLACKOUT LOCAL ORIGIN
# ============================================================

gps_x_blackout -= (
    gps_x_blackout[0]
)

gps_y_blackout -= (
    gps_y_blackout[0]
)

dr_x_blackout -= (
    dr_x_blackout[0]
)

dr_y_blackout -= (
    dr_y_blackout[0]
)

fusion_x_blackout -= (
    fusion_x_blackout[0]
)

fusion_y_blackout -= (
    fusion_y_blackout[0]
)


# ============================================================
# BLACKOUT METRICS
# ============================================================

dr_blackout_metrics = calculate_metrics(
    gps_x_blackout,
    gps_y_blackout,
    dr_x_blackout,
    dr_y_blackout,
)

fusion_blackout_metrics = calculate_metrics(
    gps_x_blackout,
    gps_y_blackout,
    fusion_x_blackout,
    fusion_y_blackout,
)


# ============================================================
# BLACKOUT DISTANCES
# ============================================================

gps_blackout_distance = (
    trajectory_distance(
        gps_x_blackout,
        gps_y_blackout,
    )
)

dr_blackout_distance = (
    trajectory_distance(
        dr_x_blackout,
        dr_y_blackout,
    )
)

fusion_blackout_distance = (
    trajectory_distance(
        fusion_x_blackout,
        fusion_y_blackout,
    )
)


# ============================================================
# SAMPLE-LEVEL ERRORS
# ============================================================

dr_error = np.sqrt(
    (dr_x - gps_x) ** 2
    +
    (dr_y - gps_y) ** 2
)

fusion_error = np.sqrt(
    (fusion_x - gps_x) ** 2
    +
    (fusion_y - gps_y) ** 2
)


# ============================================================
# ADD RESULTS TO DATAFRAME
# ============================================================

merged["gps_x_m"] = gps_x
merged["gps_y_m"] = gps_y

merged["dr_x_aligned_m"] = dr_x
merged["dr_y_aligned_m"] = dr_y

merged["fusion_x_aligned_m"] = fusion_x
merged["fusion_y_aligned_m"] = fusion_y

merged["dr_position_error_m"] = dr_error

merged["fusion_position_error_m"] = fusion_error

merged["gnss_blackout"] = (
    (merged["timestamp_ms"] >= BLACKOUT_START)
    &
    (merged["timestamp_ms"] <= BLACKOUT_END)
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"GPS distance       : "
    f"{gps_distance:.2f} m"
)

print(
    f"DR V2 distance     : "
    f"{dr_distance:.2f} m"
)

print(
    f"Fusion V1 distance : "
    f"{fusion_distance:.2f} m"
)


print_metrics(
    "DR V2",
    dr_metrics,
)

print_metrics(
    "Fusion V1",
    fusion_metrics,
)


# ============================================================
# PRINT BLACKOUT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BLACKOUT RESULTS")
print("=" * 60)

print(
    f"Blackout start     : "
    f"{BLACKOUT_START} ms"
)

print(
    f"Blackout end       : "
    f"{BLACKOUT_END} ms"
)

print(
    f"Blackout samples   : "
    f"{blackout_count}"
)

print(
    f"GPS distance       : "
    f"{gps_blackout_distance:.2f} m"
)

print(
    f"DR V2 distance     : "
    f"{dr_blackout_distance:.2f} m"
)

print(
    f"Fusion V1 distance : "
    f"{fusion_blackout_distance:.2f} m"
)


print_metrics(
    "DR V2 blackout",
    dr_blackout_metrics,
)

print_metrics(
    "Fusion V1 blackout",
    fusion_blackout_metrics,
)


# ============================================================
# SAVE SAMPLE-LEVEL RESULTS
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

lines = []

lines.append(
    "NAVIGATION EVALUATION\n"
)

lines.append(
    "=" * 60 + "\n"
)

lines.append(
    "\nOVERALL\n"
)

lines.append(
    f"Matched samples: {len(merged)}\n"
)

lines.append(
    f"GPS distance: "
    f"{gps_distance:.2f} m\n"
)

lines.append(
    f"DR V2 distance: "
    f"{dr_distance:.2f} m\n"
)

lines.append(
    f"Fusion V1 distance: "
    f"{fusion_distance:.2f} m\n"
)


lines.append(
    "\nDR V2 metrics:\n"
)

for key, value in dr_metrics.items():

    lines.append(
        f"{key}: {value}\n"
    )


lines.append(
    "\nFusion V1 metrics:\n"
)

for key, value in fusion_metrics.items():

    lines.append(
        f"{key}: {value}\n"
    )


lines.append(
    "\nBLACKOUT\n"
)

lines.append(
    f"Blackout start: "
    f"{BLACKOUT_START} ms\n"
)

lines.append(
    f"Blackout end: "
    f"{BLACKOUT_END} ms\n"
)

lines.append(
    f"Blackout samples: "
    f"{blackout_count}\n"
)

lines.append(
    f"GPS blackout distance: "
    f"{gps_blackout_distance:.2f} m\n"
)

lines.append(
    f"DR V2 blackout distance: "
    f"{dr_blackout_distance:.2f} m\n"
)

lines.append(
    f"Fusion V1 blackout distance: "
    f"{fusion_blackout_distance:.2f} m\n"
)


lines.append(
    "\nDR V2 blackout metrics:\n"
)

for key, value in dr_blackout_metrics.items():

    lines.append(
        f"{key}: {value}\n"
    )


lines.append(
    "\nFusion V1 blackout metrics:\n"
)

for key, value in fusion_blackout_metrics.items():

    lines.append(
        f"{key}: {value}\n"
    )


METRICS_FILE.write_text(
    "".join(lines),
    encoding="utf-8",
)


# ============================================================
# COMPLETE
# ============================================================

print("\nFiles saved:")

print(
    OUTPUT_FILE
)

print(
    METRICS_FILE
)

print("\n" + "=" * 60)
print("NAVIGATION EVALUATION COMPLETE")
print("=" * 60)