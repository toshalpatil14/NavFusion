import pandas as pd
from pathlib import Path

FILE = Path(r"F:\NavFusion\results\heading_interface.csv")
OUT = Path(r"F:\NavFusion\results\heading_metrics.txt")

df = pd.read_csv(FILE)

text = []

text.append("HEADING & MOTION MODULE V1\n")
text.append("="*40 + "\n\n")

text.append(f"Samples: {len(df)}\n\n")

text.append("Heading\n")
text.append(f"Min : {df.heading_deg.min():.2f}\n")
text.append(f"Max : {df.heading_deg.max():.2f}\n")
text.append(f"Mean: {df.heading_deg.mean():.2f}\n\n")

text.append("Yaw Rate\n")
text.append(f"Mean: {df.yaw_rate.mean():.4f}\n")
text.append(f"Std : {df.yaw_rate.std():.4f}\n\n")

text.append("Motion Distribution\n")

counts = df.motion_state.value_counts()

for k, v in counts.items():
    pct = v/len(df)*100
    text.append(f"{k}: {v} ({pct:.2f}%)\n")

with open(OUT, "w") as f:
    f.writelines(text)

print("Saved:", OUT)
print("".join(text))