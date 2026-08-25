import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_interface_10hz.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_gnss_blackout.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Use a middle portion of the drive for the first controlled test.
BLACKOUT_START_FRACTION = 0.40
BLACKOUT_END_FRACTION = 0.50


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("GNSS BLACKOUT EXPERIMENT SETUP")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")


# ============================================================
# CREATE GNSS AVAILABILITY FLAG
# ============================================================

n = len(df)

start_index = int(
    n * BLACKOUT_START_FRACTION
)

end_index = int(
    n * BLACKOUT_END_FRACTION
)

df["gnss_available"] = True

df.loc[
    start_index:end_index - 1,
    "gnss_available"
] = False


# ============================================================
# BLACKOUT INFORMATION
# ============================================================

blackout = df.loc[
    start_index:end_index - 1
]

start_timestamp = int(
    blackout["timestamp_ms"].iloc[0]
)

end_timestamp = int(
    blackout["timestamp_ms"].iloc[-1]
)

duration_seconds = (
    end_timestamp - start_timestamp
) / 1000.0


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GNSS BLACKOUT CREATED")
print("=" * 60)

print(
    f"Blackout start index : {start_index}"
)

print(
    f"Blackout end index   : {end_index - 1}"
)

print(
    f"Blackout start time  : "
    f"{start_timestamp} ms"
)

print(
    f"Blackout end time    : "
    f"{end_timestamp} ms"
)

print(
    f"Blackout duration    : "
    f"{duration_seconds:.2f} s"
)

print(
    f"GNSS available rows  : "
    f"{df['gnss_available'].sum()}"
)

print(
    f"GNSS blackout rows   : "
    f"{(~df['gnss_available']).sum()}"
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)