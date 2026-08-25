import pandas as pd
from pathlib import Path

FILE = Path(r"F:\NavFusion\results\heading_motion_output.csv")

df = pd.read_csv(FILE)

print("="*40)
print("HEADING MODULE EVALUATION")
print("="*40)

print("\nSamples:", len(df))

print("\nHeading statistics")
print(df["heading_deg"].describe().round(2))

print("\nYaw-rate statistics")
print(df["yaw_rate"].describe().round(3))

print("\nMotion distribution")
counts = df["motion_state"].value_counts()
percent = (counts/len(df)*100).round(2)

for s in counts.index:
    print(f"{s:10s}: {counts[s]:6d} ({percent[s]:5.2f}%)")

print("\nAverage speed by motion state")
print(df.groupby("motion_state")["GPS SPEED (Kmh)"].mean().round(2))