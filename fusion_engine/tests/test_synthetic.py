import sys
from pathlib import Path

# Add src folder to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from blackout import is_gnss_available
from fusion_engine import FusionEngine
from models import NavigationInput
from synthetic_data import generate_vehicle_data


def run_noisy_navigation_test():

    origin_lat = 19.9975
    origin_lon = 73.7898

    engine = FusionEngine(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    dt = 0.1
    total_time = 40.0

    # True vehicle motion
    true_speed_mps = 10.0
    true_heading_rad = 0.0

    print("=== NOISY DEAD RECKONING TEST ===\n")

    t = 0.0
    step = 0

    while t <= total_time:

        # Generate true motion and noisy measurements
        data = generate_vehicle_data(
            timestamp=t,
            speed_mps=true_speed_mps,
            heading_rad=true_heading_rad,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

        # Check GNSS availability
        gnss_available = is_gnss_available(t)

        # Remove GNSS during blackout
        if gnss_available:
            gnss_lat = data["true_lat"]
            gnss_lon = data["true_lon"]
        else:
            gnss_lat = None
            gnss_lon = None

        # Send noisy measurements to navigation engine
        nav_input = NavigationInput(
            timestamp=t,
            speed_mps=data["noisy_speed"],
            heading_rad=data["noisy_heading"],
            gnss_lat=gnss_lat,
            gnss_lon=gnss_lon,
            gnss_available=gnss_available,
        )

        # Get estimated navigation state
        state = engine.update(nav_input)

        # Print every 5 seconds
        if step % 50 == 0:

            error_x = state.x_m - data["true_x"]
            error_y = state.y_m - data["true_y"]

            position_error = (
                error_x ** 2 + error_y ** 2
            ) ** 0.5

            print(
                f"t={t:5.1f}s | "
                f"mode={state.mode:18s} | "
                f"true_x={data['true_x']:8.2f} | "
                f"estimated_x={state.x_m:8.2f} | "
                f"error={position_error:6.2f} m"
            )

        step += 1
        t += dt

    print("\nTEST COMPLETED")


if __name__ == "__main__":
    run_noisy_navigation_test()