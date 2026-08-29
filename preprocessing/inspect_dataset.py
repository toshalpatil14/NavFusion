import pandas as pd

# --------------------------------------------------
# IO-VNBD S1 smartphone dataset
# --------------------------------------------------

file_path = (
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "S (Driver A)/S1/S-S1.csv"
)

print("Loading dataset...")

df = pd.read_csv(file_path, encoding="latin1")
print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nShape:")
print(df.shape)

print("\nColumns:")

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())