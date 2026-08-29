import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AI_SPEED_FILE = (
    PROJECT_ROOT
    / "results"
    / "ai_speed_output.csv"
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

ai = pd.read_csv(AI_SPEED_FILE)
heading = pd.read_csv(HEADING_FILE)

print("\nInput:")
print(f"AI speed samples : {len(ai)}")
print(f"Heading samples  : {len(heading)}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

ai_required = [
    "timestamp_ms",
    "ai_speed_mps",
    "speed_confidence",
]

heading_required = [
    "timestamp_ms",
    "heading_deg",
    "yaw_rate",
    "motion_state",
]

for col in ai_required:
    if col not in ai.columns:
        raise ValueError(f"Missing AI speed column: {col}")

for col in heading_required:
    if col not in heading.columns:
        raise ValueError(f"Missing heading column: {col}")


# ============================================================
# SORT
# ============================================================

ai = (
    ai[ai_required]
    .sort_values("timestamp_ms")
    .reset_index(drop=True)
)

heading = (
    heading[heading_required]
    .sort_values("timestamp_ms")
    .reset_index(drop=True)
)


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

if ai["timestamp_ms"].duplicated().any():
    raise ValueError("AI speed contains duplicate timestamps.")

if heading["timestamp_ms"].duplicated().any():
    raise ValueError("Heading contains duplicate timestamps.")

if not ai["timestamp_ms"].is_monotonic_increasing:
    raise ValueError("AI timestamps are not ordered.")

if not heading["timestamp_ms"].is_monotonic_increasing:
    raise ValueError("Heading timestamps are not ordered.")


# ============================================================
# NUMERIC VALIDATION
# ============================================================

for col in ["ai_speed_mps", "speed_confidence"]:
    ai[col] = pd.to_numeric(ai[col], errors="coerce")

for col in ["heading_deg", "yaw_rate"]:
    heading[col] = pd.to_numeric(heading[col], errors="coerce")

if ai[["ai_speed_mps", "speed_confidence"]].isna().any().any():
    raise ValueError("AI speed contains NaN values.")

if heading[["heading_deg", "yaw_rate"]].isna().any().any():
    raise ValueError("Heading contains NaN values.")

if (ai["ai_speed_mps"] < 0).any():
    raise ValueError("AI speed contains negative values.")


# ============================================================
# MASTER TIMELINE = 10-HZ HEADING TIMELINE
# ============================================================

# AI predictions are approximately 1 Hz.
# The heading/IMU timeline is approximately 10 Hz.
#
# Therefore:
#   - heading provides the master navigation timestamps
#   - latest available AI prediction is carried forward
#     onto each 10-Hz navigation sample

navigation = pd.merge_asof(
    heading,
    ai,
    on="timestamp_ms",
    direction="backward",
)


# ============================================================
# REMOVE SAMPLES BEFORE FIRST AI PREDICTION
# ============================================================

navigation = navigation.dropna(
    subset=["ai_speed_mps", "speed_confidence"]
).reset_index(drop=True)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

navigation = navigation[
    [
        "timestamp_ms",
        "ai_speed_mps",
        "speed_confidence",
        "heading_deg",
        "yaw_rate",
        "motion_state",
    ]
]


# ============================================================
# FINAL VALIDATION
# ============================================================

if navigation.empty:
    raise ValueError("Navigation interface is empty.")

if navigation["timestamp_ms"].duplicated().any():
    raise ValueError("Navigation interface contains duplicate timestamps.")

if not navigation["timestamp_ms"].is_monotonic_increasing:
    raise ValueError("Navigation timestamps are not ordered.")

if navigation.isna().any().any():
    raise ValueError("Navigation interface contains NaN values.")


# Check that we really retained approximately 10-Hz timing.
dt = navigation["timestamp_ms"].diff().dropna() / 1000.0

print("\n" + "=" * 60)
print("NAVIGATION INTERFACE CREATED")
print("=" * 60)

print(f"Rows            : {len(navigation)}")
print(f"First timestamp : {navigation['timestamp_ms'].iloc[0]}")
print(f"Last timestamp  : {navigation['timestamp_ms'].iloc[-1]}")

print(
    f"dt median       : {dt.median():.3f} s"
)

print(
    f"dt min/max      : "
    f"{dt.min():.3f} / {dt.max():.3f} s"
)

print(
    f"AI speed mean   : "
    f"{navigation['ai_speed_mps'].mean():.4f} m/s"
)

print(
    f"AI confidence   : "
    f"{navigation['speed_confidence'].mean():.4f}"
)

print("\nColumns:")
print(navigation.columns.tolist())

print("\nFirst 5 rows:")
print(navigation.head())


# ============================================================
# SAVE
# ============================================================

navigation.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved to:")
print(OUTPUT_FILE)