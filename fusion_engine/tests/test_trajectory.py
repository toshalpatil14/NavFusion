import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
RESULTS_PATH = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_PATH))

# Create results folder if needed
RESULTS_PATH.mkdir(exist_ok=True)

from blackout import is_gnss_available
from fusion_engine import FusionEngine
from models import NavigationInput
from synthetic_data import generate_vehicle_data
from metrics import calculate_metrics, print_metrics_report


def run_trajectory_test():

    # Starting location
    origin_lat = 19.9975
    origin_lon = 73.7898

    # Create fusion engine
    engine = FusionEngine(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    # Simulation settings
    dt = 0.1
    total_time = 40.0

    true_speed_mps = 10.0
    true_heading_rad = 0.0

    # Lists for graph
    true_x_values = []
    true_y_values = []

    estimated_x_values = []
    estimated_y_values = []

    blackout_x_values = []
    blackout_y_values = []
    blackout_mask = []

    # Run simulation
    t = 0.0

    while t <= total_time:

        data = generate_vehicle_data(
            timestamp=t,
            speed_mps=true_speed_mps,
            heading_rad=true_heading_rad,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

        # Check GNSS
        gnss_available = is_gnss_available(t)
        blackout_mask.append(not gnss_available)

        if gnss_available:
            gnss_lat = data["true_lat"]
            gnss_lon = data["true_lon"]
        else:
            gnss_lat = None
            gnss_lon = None

        # Create input
        nav_input = NavigationInput(
            timestamp=t,
            speed_mps=data["noisy_speed"],
            heading_rad=data["noisy_heading"],
            gnss_lat=gnss_lat,
            gnss_lon=gnss_lon,
            gnss_available=gnss_available,
        )

        # Run EKF fusion engine
        state = engine.update(nav_input)

        # Store ground truth
        true_x_values.append(data["true_x"])
        true_y_values.append(data["true_y"])

        # Store estimated trajectory
        estimated_x_values.append(state.x_m)
        estimated_y_values.append(state.y_m)

        # Store blackout zone
        if not gnss_available:
            blackout_x_values.append(data["true_x"])
            blackout_y_values.append(data["true_y"])

        t += dt
    # ----------------------------------------------
    # CALCULATE PERFORMANCE METRICS
    # ----------------------------------------------

    metrics = calculate_metrics(
        true_x_values=true_x_values,
        true_y_values=true_y_values,
        estimated_x_values=estimated_x_values,
        estimated_y_values=estimated_y_values,
        blackout_mask=blackout_mask,
    )

    print_metrics_report(metrics)
    # Create graph
    plt.figure(figsize=(12, 6))

    # Ground truth path
    plt.plot(
        true_x_values,
        true_y_values,
        label="Ground Truth",
        linewidth=2,
    )

    # EKF estimated path
    plt.plot(
        estimated_x_values,
        estimated_y_values,
        label="EKF Fusion Estimate",
        linewidth=2,
        linestyle="--",
    )

    # GNSS blackout section
    plt.scatter(
        blackout_x_values,
        blackout_y_values,
        label="GNSS Blackout",
        s=12,
    )

    # Start point
    plt.scatter(
        true_x_values[0],
        true_y_values[0],
        label="Start",
        s=80,
        marker="o",
    )

    # End point
    plt.scatter(
        true_x_values[-1],
        true_y_values[-1],
        label="End",
        s=80,
        marker="X",
    )

    # Labels
    plt.title("GNSS + INS EKF Fusion Trajectory")
    plt.xlabel("East Position (m)")
    plt.ylabel("North Position (m)")

    plt.legend()
    plt.grid(True)

    # Save graph
    output_file = RESULTS_PATH / "trajectory.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    print("TRAJECTORY TEST COMPLETED")
    print(f"Graph saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    run_trajectory_test()