from config import (
    DEFAULT_DT,
    MODE_GNSS_INS,
    MODE_DEAD_RECKONING,
)

from coordinates import latlon_to_xy, xy_to_latlon
from ekf import PositionEKF
from models import NavigationInput, NavigationState


class FusionEngine:
    """
    GNSS + INS Fusion Engine using an Extended Kalman Filter.

    GNSS available:
        1. Predict position using noisy speed + heading
        2. Correct prediction using GNSS

    GNSS unavailable:
        1. Predict position using speed + heading
        2. No GNSS correction
        3. Dead reckoning continues
    """

    def __init__(self, origin_lat: float, origin_lon: float):

        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

        # Create EKF
        self.ekf = PositionEKF(
            initial_x=0.0,
            initial_y=0.0,
            process_noise=0.5,
            measurement_noise=4.0,
        )

        # Vehicle state
        self.speed_mps = 0.0
        self.heading_rad = 0.0

        # Time tracking
        self.last_timestamp = None

        # Initialization flag
        self.initialized = False

    def update(self, nav_input: NavigationInput) -> NavigationState:
        """
        Process one navigation measurement.
        """

        # -----------------------------------
        # CALCULATE TIME DIFFERENCE
        # -----------------------------------

        if self.last_timestamp is None:
            dt = DEFAULT_DT
        else:
            dt = (
                nav_input.timestamp
                - self.last_timestamp
            )

            if dt <= 0:
                dt = DEFAULT_DT

        # Store current vehicle motion
        self.speed_mps = nav_input.speed_mps
        self.heading_rad = nav_input.heading_rad

        # -----------------------------------
        # INITIALIZE FROM FIRST GNSS FIX
        # -----------------------------------

        if (
            not self.initialized
            and nav_input.gnss_available
            and nav_input.gnss_lat is not None
            and nav_input.gnss_lon is not None
        ):

            initial_x, initial_y = latlon_to_xy(
                nav_input.gnss_lat,
                nav_input.gnss_lon,
                self.origin_lat,
                self.origin_lon,
            )

            self.ekf = PositionEKF(
                initial_x=initial_x,
                initial_y=initial_y,
                process_noise=0.5,
                measurement_noise=4.0,
            )

            self.initialized = True

        # -----------------------------------
        # EKF PREDICTION
        # -----------------------------------

        self.ekf.predict(
            speed_mps=self.speed_mps,
            heading_rad=self.heading_rad,
            dt=dt,
        )

        # -----------------------------------
        # GNSS CORRECTION
        # -----------------------------------

        if (
            nav_input.gnss_available
            and nav_input.gnss_lat is not None
            and nav_input.gnss_lon is not None
        ):

            gnss_x, gnss_y = latlon_to_xy(
                nav_input.gnss_lat,
                nav_input.gnss_lon,
                self.origin_lat,
                self.origin_lon,
            )

            # Correct EKF prediction with GNSS
            self.ekf.update(
                measured_x=gnss_x,
                measured_y=gnss_y,
            )

            mode = MODE_GNSS_INS
            confidence = 0.95

        else:

            # GNSS blackout:
            # EKF continues only with inertial prediction
            mode = MODE_DEAD_RECKONING
            confidence = 0.70

        # -----------------------------------
        # GET EKF POSITION
        # -----------------------------------

        x_m, y_m = self.ekf.get_position()

        # Convert local position back to GPS
        latitude, longitude = xy_to_latlon(
            x_m,
            y_m,
            self.origin_lat,
            self.origin_lon,
        )

        # Save timestamp
        self.last_timestamp = nav_input.timestamp

        return NavigationState(
            timestamp=nav_input.timestamp,
            x_m=x_m,
            y_m=y_m,
            latitude=latitude,
            longitude=longitude,
            speed_mps=self.speed_mps,
            heading_rad=self.heading_rad,
            mode=mode,
            confidence=confidence,
        )