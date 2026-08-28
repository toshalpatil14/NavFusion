"""Timestamp-driven GNSS/INS EKF for the Day 2/3 navigation interface."""

import math

from config import MEASUREMENT_NOISE_R, MODE_DEAD_RECKONING, MODE_GNSS_INS, PROCESS_NOISE_Q
from coordinates import latlon_to_xy, xy_to_latlon
from ekf import PositionEKF
from models import NavigationInput, NavigationState


class FusionEngine:
    """Fuse AI speed and corrected compass heading with available GNSS."""

    def __init__(self, origin_lat: float, origin_lon: float, process_noise: float = PROCESS_NOISE_Q, measurement_noise: float = MEASUREMENT_NOISE_R) -> None:
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.ekf = PositionEKF(0.0, 0.0, process_accel_noise=process_noise, gnss_position_noise=measurement_noise, speed_noise=1.0)
        self.last_timestamp_ms: int | None = None
        self.initialized = False

    def update(self, nav: NavigationInput) -> NavigationState:
        if self.last_timestamp_ms is None:
            dt_s = 0.0
        else:
            dt_s = (nav.timestamp_ms - self.last_timestamp_ms) / 1000.0
            if dt_s < 0:
                raise ValueError("EKF input timestamps must be ordered.")

        if not self.initialized and nav.gnss_available and nav.gnss_lat is not None and nav.gnss_lon is not None:
            initial_x, initial_y = latlon_to_xy(nav.gnss_lat, nav.gnss_lon, self.origin_lat, self.origin_lon)
            heading_rad = math.radians((nav.heading_deg + 180.0) % 360.0)
            self.ekf = PositionEKF(
                initial_x,
                initial_y,
                nav.speed_mps * math.sin(heading_rad),
                nav.speed_mps * math.cos(heading_rad),
                process_accel_noise=self.process_noise,
                gnss_position_noise=self.measurement_noise,
                speed_noise=1.0,
            )
            self.initialized = True

        # Phone azimuth is 180 degrees opposite S-Vw4 vehicle travel.
        travel_heading_deg = (nav.heading_deg + 180.0) % 360.0
        heading_rad = math.radians(travel_heading_deg)
        if dt_s > 0.0:
            self.ekf.predict(0.0, 0.0, dt_s)
        self.ekf.update_speed(nav.speed_mps, heading_rad)

        if nav.gnss_available and nav.gnss_lat is not None and nav.gnss_lon is not None:
            measured_x, measured_y = latlon_to_xy(nav.gnss_lat, nav.gnss_lon, self.origin_lat, self.origin_lon)
            self.ekf.update_gnss(measured_x, measured_y)
            mode = MODE_GNSS_INS
        else:
            mode = MODE_DEAD_RECKONING

        x_m, y_m = self.ekf.position()
        latitude, longitude = xy_to_latlon(x_m, y_m, self.origin_lat, self.origin_lon)
        self.last_timestamp_ms = nav.timestamp_ms
        return NavigationState(nav.timestamp_ms, dt_s, x_m, y_m, latitude, longitude, mode)
