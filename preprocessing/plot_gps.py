import pandas as pd
import matplotlib.pyplot as plt

file_path = (
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "S (Driver A)/S1/S-S1.csv"
)

df = pd.read_csv(file_path, encoding="latin1")

lat = df["GPS LATITUDE (degrees)"]
lon = df[" GPS LONGITUDE (degrees)"]

plt.figure(figsize=(10, 8))

plt.plot(lon, lat)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("IO-VNBD S1 — GPS Trajectory")

plt.grid(True)
plt.axis("equal")

plt.show()