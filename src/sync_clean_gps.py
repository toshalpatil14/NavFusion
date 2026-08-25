import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NAV_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_interface_10hz.csv"
)

GPS_FILE = (
    PROJECT_ROOT
    / "results"
    / "clean_gps_reference.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gps_reference_10hz.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("CLEAN GPS + NAVIGATION SYNCHRONIZATION")
print("=" * 60)

nav = pd.read_csv(NAV_FILE)
gps = pd.read_csv(GPS_FILE)

print("\nInput datasets:")
print(f"Navigation rows : {len(nav)}")
print(f"GPS rows        : {len(gps)}")


# ============================================================
# KEEP VALID GPS
# ============================================================

gps = gps[
    gps["gps_valid"] == True
].copy()


# ============================================================
# RENAME TIMESTAMP
# ============================================================

gps = gps.rename(
    columns={
        "TIME SINCE START (ms)": "timestamp_ms",
        "GPS LATITUDE (degrees)": "gps_latitude",
        "GPS LONGITUDE (degrees)": "gps_longitude",
        "GPS SPEED (Kmh)": "gps_speed_kmh",
    }
)


# ============================================================
# EXACT TIMESTAMP MATCH
# ============================================================

merged = pd.merge(
    nav,
    gps[
        [
            "timestamp_ms",
            "gps_latitude",
            "gps_longitude",
            "gps_speed_kmh",
            "gps_jump_m",
        ]
    ],
    on="timestamp_ms",
    how="left",
)


# ============================================================
# MATCH STATISTICS
# ============================================================

matched = (
    merged["gps_latitude"].notna()
)

matched_count = matched.sum()
unmatched_count = (~matched).sum()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SYNCHRONIZATION RESULTS")
print("=" * 60)

print(
    f"Navigation rows      : {len(nav)}"
)

print(
    f"Clean GPS rows       : {len(gps)}"
)

print(
    f"Matched navigation   : {matched_count}"
)

print(
    f"Unmatched navigation : {unmatched_count}"
)

print(
    f"Match rate           : "
    f"{100 * matched_count / len(nav):.2f}%"
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFirst 5 rows:")

print(
    merged.head().to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)
print("CLEAN GPS SYNCHRONIZATION COMPLETE")
print("=" * 60)