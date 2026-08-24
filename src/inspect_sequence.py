import pandas as pd
from pathlib import Path

DATA_DIR = Path(
    r"D:\IDR-AI\data\raw\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset"
    r"\S-Dataset"
)

FILES = [
    "S-Vw4.csv",
    "S-S4.csv",
    "S-S2.csv",
    "S-M.csv",
    "S-Vfa02.csv",
    "S-Vw2.csv",
]

for filename in FILES:

    path = DATA_DIR / filename

    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()

    time = pd.to_numeric(
        df["TIME SINCE START (ms)"],
        errors="coerce"
    )

    speed = pd.to_numeric(
        df["GPS SPEED (Kmh)"],
        errors="coerce"
    )

    print("\n================================")
    print(filename)
    print("================================")

    print("Rows:", len(df))

    print("\nTIME:")
    print("First:", time.iloc[0])
    print("Last:", time.iloc[-1])
    print("Difference:", time.iloc[-1] - time.iloc[0])

    print("\nTIME DIFFERENCES:")
    print(time.diff().describe())

    print("\nSPEED:")
    print("Min:", speed.min())
    print("Max:", speed.max())
    print("Mean:", speed.mean())
    print("Median:", speed.median())

    print("\nFIRST 20 SPEED VALUES:")
    print(speed.head(20).to_list())

    print("\nLAST 20 SPEED VALUES:")
    print(speed.tail(20).to_list())