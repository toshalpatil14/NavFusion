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
    / "speed_heading_interface.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("SPEED + HEADING SYNCHRONIZATION")
print("=" * 60)

speed = pd.read_csv(SPEED_FILE)
heading = pd.read_csv(HEADING_FILE)

print("\nInput datasets:")
print(f"Speed predictions : {len(speed)}")
print(f"Heading samples   : {len(heading)}")


# ============================================================
# VALIDATE COLUMNS
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

for column in speed_required:

    if column not in speed.columns:
        raise ValueError(
            f"Missing speed column: {column}"
        )

for column in heading_required:

    if column not in heading.columns:
        raise ValueError(
            f"Missing heading column: {column}"
        )


# ============================================================
# EXACT TIMESTAMP MERGE
# ============================================================

merged = pd.merge(
    speed[speed_required],
    heading[heading_required],
    on="timestamp_ms",
    how="inner",
)


# ============================================================
# SPEED CONVERSION
# ============================================================

merged["speed_mps"] = (
    merged["estimated_speed_kmh"] / 3.6
)


# ============================================================
# COLUMN ORDER
# ============================================================

merged = merged[
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

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SYNCHRONIZATION RESULTS")
print("=" * 60)

print(
    f"Speed predictions : {len(speed)}"
)

print(
    f"Heading samples   : {len(heading)}"
)

print(
    f"Merged rows       : {len(merged)}"
)

print(
    f"Match rate        : "
    f"{100 * len(merged) / len(speed):.2f}%"
)

print(
    f"First timestamp   : "
    f"{merged['timestamp_ms'].iloc[0]}"
)

print(
    f"Last timestamp    : "
    f"{merged['timestamp_ms'].iloc[-1]}"
)

print("\nFirst 10 rows:")

print(
    merged.head(10).to_string(index=False)
)

print("\nColumns:")

print(
    list(merged.columns)
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 60)