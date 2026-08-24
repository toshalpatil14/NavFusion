import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(
    r"D:\IDR-AI\data\raw\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Uncategorised IOVNB Dataset"
    r"\S-Dataset"
    r"\S-Vw4.csv"
)

OUTPUT_DIR = Path(r"D:\IDR-AI\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 Hz × 5 seconds
WINDOW_SIZE = 50

# Step between windows
# 10 samples = 1 second
STRIDE = 10

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GYROSCOPE X (rad/s)",
    "GYROSCOPE Y (rad/s)",
    "GYROSCOPE Z (rad/s)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",
]

TARGET_COLUMN = "GPS SPEED (Kmh)"

# ============================================================
# LOAD
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    encoding="latin1"
)

df.columns = df.columns.str.strip()

print("Rows:", len(df))

# ============================================================
# CLEAN NUMERICAL DATA
# ============================================================

columns_needed = FEATURE_COLUMNS + [
    TARGET_COLUMN,
    "TIME SINCE START (ms)"
]

df = df[columns_needed].copy()

for column in columns_needed:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove invalid rows
df = df.dropna().reset_index(drop=True)

print("Rows after cleaning:", len(df))

# ============================================================
# TIME QUALITY
# ============================================================

time = df["TIME SINCE START (ms)"].values

dt = np.diff(time)

print("\nTime statistics:")
print("Median dt:", np.median(dt), "ms")
print("Mean dt:", np.mean(dt), "ms")

# ============================================================
# TIME-BASED SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nSplit:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# NORMALIZATION
# IMPORTANT:
# Fit scaler ONLY on training data.
# ============================================================

scaler = StandardScaler()

scaler.fit(
    train_df[FEATURE_COLUMNS]
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "imu_scaler.pkl"
)

# ============================================================
# WINDOW FUNCTION
# ============================================================

def create_windows(dataframe):

    X_raw = dataframe[FEATURE_COLUMNS].values
    y_raw = dataframe[TARGET_COLUMN].values

    X = []
    y = []

    for start in range(
        0,
        len(dataframe) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        window = X_raw[start:end]

        # Target = speed at END of window
        target = y_raw[end - 1]

        X.append(window)
        y.append(target)

    return np.array(X, dtype=np.float32), \
           np.array(y, dtype=np.float32)


# ============================================================
# CREATE WINDOWS
# ============================================================

X_train, y_train = create_windows(train_df)
X_val, y_val = create_windows(val_df)
X_test, y_test = create_windows(test_df)

# ============================================================
# NORMALIZE WINDOWS
# ============================================================

def normalize_windows(X):

    original_shape = X.shape

    X = X.reshape(-1, X.shape[-1])

    X = scaler.transform(X)

    return X.reshape(original_shape).astype(np.float32)


X_train = normalize_windows(X_train)
X_val = normalize_windows(X_val)
X_test = normalize_windows(X_test)

# ============================================================
# SAVE
# ============================================================

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_val.npy", y_val)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("WINDOW DATASET CREATED")
print("==============================")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nSpeed target:")
print("Train min:", y_train.min())
print("Train max:", y_train.max())
print("Train mean:", y_train.mean())

print("\nSaved to:")
print(OUTPUT_DIR)