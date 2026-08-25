import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPEED_FILE = (
    PROJECT_ROOT
    / "results"
    / "S-Vw4_speed_predictions.csv"
)

HEADING_FILE = (
    PROJECT_ROOT
    / "results"
    / "heading_motion_output.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_interface_10hz.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("10-HZ NAVIGATION INTERFACE")
print("=" * 60)

speed = pd.read_csv(SPEED_FILE)
heading = pd.read_csv(HEADING_FILE)

print("\nInput:")
print(f"Speed predictions : {len(speed)}")
print(f"Heading samples   : {len(heading)}")


# ============================================================
# VALIDATE
# ============================================================

speed_required = [
    "timestamp_ms",
    "estimated_speed_kmh",
]

heading_required = [
    "timestamp_ms",
    "heading_deg",
    "yaw_rate",
    "motion_state",
]

for col in speed_required:
    if col not in speed.columns:
        raise ValueError(
            f"Missing speed column: {col}"
        )

for col in heading_required:
    if col not in heading.columns:
        raise ValueError(
            f"Missing heading column: {col}"
        )


# ============================================================
# SORT
# ============================================================

speed = speed[
    speed_required
].sort_values(
    "timestamp_ms"
).reset_index(drop=True)

heading = heading[
    heading_required
].sort_values(
    "timestamp_ms"
).reset_index(drop=True)


# ============================================================
# MERGE SPEED ONTO 10-HZ HEADING TIMELINE
# ============================================================

navigation = pd.merge_asof(
    heading,
    speed,
    on="timestamp_ms",
    direction="backward",
)


# ============================================================
# REMOVE HEADING SAMPLES BEFORE FIRST SPEED PREDICTION
# ============================================================

navigation = navigation.dropna(
    subset=["estimated_speed_kmh"]
).reset_index(drop=True)


# ============================================================
# SPEED CONVERSION
# ============================================================

navigation["speed_mps"] = (
    navigation["estimated_speed_kmh"]
    / 3.6
)


# ============================================================
# COLUMN ORDER
# ============================================================

navigation = navigation[
    [
        "timestamp_ms",
        "estimated_speed_kmh",
        "speed_mps",
        "heading_deg",
        "yaw_rate",
        "motion_state",
    ]
]


# ============================================================
# SAVE
# ============================================================

navigation.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NAVIGATION INTERFACE CREATED")
print("=" * 60)

print(
    f"Navigation rows : "
    f"{len(navigation)}"
)

print(
    f"First timestamp : "
    f"{navigation['timestamp_ms'].iloc[0]}"
)

print(
    f"Last timestamp  : "
    f"{navigation['timestamp_ms'].iloc[-1]}"
)

print(
    f"Mean speed      : "
    f"{navigation['estimated_speed_kmh'].mean():.3f} km/h"
)

print("\nFirst 10 rows:")

print(
    navigation.head(10)
    .to_string(index=False)
)

print("\nColumns:")

print(
    list(navigation.columns)
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 60)