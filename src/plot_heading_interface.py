import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(r"F:\NavFusion\results\heading_interface.csv")
OUT = Path(r"F:\NavFusion\results")

df = pd.read_csv(FILE)

# Heading
plt.figure(figsize=(12,4))
plt.plot(df.heading_deg, linewidth=0.8)
plt.title("Heading")
plt.xlabel("Sample")
plt.ylabel("Degrees")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT/"heading_plot.png", dpi=150)
plt.close()

# Yaw rate
plt.figure(figsize=(12,4))
plt.plot(df.yaw_rate, linewidth=0.8)
plt.title("Yaw Rate")
plt.xlabel("Sample")
plt.ylabel("rad/s")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT/"yaw_rate_plot.png", dpi=150)
plt.close()

# Motion distribution
counts = df.motion_state.value_counts()

plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.title("Motion State Distribution")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(OUT/"motion_distribution.png", dpi=150)
plt.close()

print("Plots saved.")