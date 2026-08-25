import math
import random

from coordinates import xy_to_latlon


def generate_vehicle_data(
    timestamp: float,
    speed_mps: float,
    heading_rad: float,
    origin_lat: float,
    origin_lon: float,
    speed_noise_std: float = 0.5,
    heading_noise_std: float = 0.02,
    speed_bias_mps: float = 0.15,
    heading_bias_rad: float = 0.01,
):
    """
    Generate synthetic vehicle navigation data.

    Simulates:
    - True vehicle motion
    - Random sensor noise
    - Persistent sensor bias

    Persistent bias is important because it causes
    dead-reckoning drift over time.
    """

    # ----------------------------------------
    # TRUE VEHICLE POSITION
    # ----------------------------------------

    true_x = (
        speed_mps
        * math.cos(heading_rad)
        * timestamp
    )

    true_y = (
        speed_mps
        * math.sin(heading_rad)
        * timestamp
    )

    # Convert true position to latitude/longitude
    true_lat, true_lon = xy_to_latlon(
        true_x,
        true_y,
        origin_lat,
        origin_lon,
    )

    # ----------------------------------------
    # SENSOR MEASUREMENTS
    # ----------------------------------------

    # Persistent bias + random noise
    noisy_speed = (
        speed_mps
        + speed_bias_mps
        + random.gauss(0.0, speed_noise_std)
    )

    noisy_heading = (
        heading_rad
        + heading_bias_rad
        + random.gauss(0.0, heading_noise_std)
    )

    return {
        "true_x": true_x,
        "true_y": true_y,

        "true_lat": true_lat,
        "true_lon": true_lon,

        "noisy_speed": noisy_speed,
        "noisy_heading": noisy_heading,
    }