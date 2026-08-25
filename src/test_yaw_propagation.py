import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "navigation_interface_10hz.csv"
)


df = pd.read_csv(INPUT_FILE)

heading = df["heading_deg"].to_numpy()
yaw = df["yaw_rate"].to_numpy()
timestamp = df["timestamp_ms"].to_numpy()

dt = np.diff(timestamp) / 1000.0

# Compare two possible conventions:
# 1. heading + yaw * dt
# 2. heading - yaw * dt

h_plus = np.empty(len(df))
h_minus = np.empty(len(df))

h_plus[0] = heading[0]
h_minus[0] = heading[0]

for i in range(1, len(df)):
    h_plus[i] = (
        h_plus[i - 1]
        + yaw[i] * dt[i - 1]
    ) % 360.0

    h_minus[i] = (
        h_minus[i - 1]
        - yaw[i] * dt[i - 1]
    ) % 360.0


def circular_error(a, b):
    e = np.abs(a - b)
    return np.minimum(e, 360.0 - e)


error_plus = circular_error(
    h_plus,
    heading
)

error_minus = circular_error(
    h_minus,
    heading
)


print("=" * 60)
print("YAW-RATE PROPAGATION TEST")
print("=" * 60)

print(f"Samples: {len(df)}")

print()
print("HEADING + YAW * DT")
print(
    f"Mean error   : {error_plus.mean():.4f} deg"
)
print(
    f"Median error : {np.median(error_plus):.4f} deg"
)
print(
    f"P90 error    : {np.percentile(error_plus, 90):.4f} deg"
)
print(
    f"Max error    : {error_plus.max():.4f} deg"
)

print()
print("HEADING - YAW * DT")
print(
    f"Mean error   : {error_minus.mean():.4f} deg"
)
print(
    f"Median error : {np.median(error_minus):.4f} deg"
)
print(
    f"P90 error    : {np.percentile(error_minus, 90):.4f} deg"
)
print(
    f"Max error    : {error_minus.max():.4f} deg"
)

print("=" * 60)