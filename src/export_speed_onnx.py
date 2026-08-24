import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path(
    r"D:\IDR-AI\models\speed_cnn_v1.pt"
)

OUTPUT_FILE = Path(
    r"D:\IDR-AI\models\speed_cnn_v1.onnx"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==============================")
print("SPEED CNN → ONNX EXPORT")
print("==============================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL
# ============================================================

class SpeedCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                9,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.regressor(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading PyTorch model...")

model = SpeedCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# DUMMY INPUT
# ============================================================

# Model expects:
#
# Batch × Channels × Time
#
# Batch = 1
# Channels = 9
# Time = 50

dummy_input = torch.randn(
    1,
    9,
    50,
    device=device
)


print("\nDummy input shape:")
print(dummy_input.shape)


# ============================================================
# PYTORCH REFERENCE OUTPUT
# ============================================================

with torch.no_grad():

    pytorch_output = model(
        dummy_input
    )

print(
    "\nPyTorch output:",
    pytorch_output.item()
)


# ============================================================
# EXPORT
# ============================================================

print("\nExporting ONNX model...")

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_FILE,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["imu_window"],
    output_names=["predicted_speed"],
    dynamic_axes={
        "imu_window": {
            0: "batch"
        },
        "predicted_speed": {
            0: "batch"
        }
    }
)

print(
    "ONNX model saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# CHECK ONNX MODEL
# ============================================================

print("\nChecking ONNX model...")

onnx_model = onnx.load(
    OUTPUT_FILE
)

onnx.checker.check_model(
    onnx_model
)

print(
    "ONNX model structure: OK"
)


# ============================================================
# ONNX RUNTIME
# ============================================================

print("\nRunning ONNX Runtime...")

session = ort.InferenceSession(
    str(OUTPUT_FILE),
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name


# ONNX Runtime uses CPU here.
# This is intentional for compatibility testing.

onnx_output = session.run(
    [output_name],
    {
        input_name:
            dummy_input
            .detach()
            .cpu()
            .numpy()
    }
)[0]


onnx_prediction = float(
    onnx_output[0][0]
)


print(
    "ONNX output:",
    onnx_prediction
)


# ============================================================
# NUMERICAL COMPARISON
# ============================================================

difference = abs(
    pytorch_output.item()
    -
    onnx_prediction
)


print("\n==============================")
print("PYTORCH vs ONNX")
print("==============================")

print(
    f"PyTorch : {pytorch_output.item():.8f}"
)

print(
    f"ONNX    : {onnx_prediction:.8f}"
)

print(
    f"Difference : {difference:.10f}"
)


# ============================================================
# PASS / FAIL
# ============================================================

if difference < 1e-4:

    print(
        "\nPASS: ONNX matches PyTorch."
    )

else:

    print(
        "\nWARNING: ONNX difference is larger than expected."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n==============================")
print("ONNX MODEL INFORMATION")
print("==============================")

print(
    "Input:",
    session.get_inputs()[0].shape
)

print(
    "Output:",
    session.get_outputs()[0].shape
)

print(
    "Providers:",
    session.get_providers()
)

print("\n==============================")
print("EXPORT COMPLETE")
print("==============================")