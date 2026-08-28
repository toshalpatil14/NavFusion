import numpy as np


class PositionEKF:
    """
    Simple 2D Extended Kalman Filter.

    State:
        x = [position_x, position_y]

    Prediction:
        Uses vehicle speed and heading.

    Correction:
        Uses GNSS position when available.
    """

    def __init__(
        self,
        initial_x=0.0,
        initial_y=0.0,
        process_noise=1.0,
        measurement_noise=5.0,
    ):

        # State vector:
        # [x_position, y_position]
        self.x = np.array(
            [
                [initial_x],
                [initial_y],
            ],
            dtype=float,
        )

        # State covariance matrix
        self.P = np.eye(2) * 10.0

        # Process noise covariance
        self.Q = np.eye(2) * process_noise

        # GNSS measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        # Measurement matrix
        self.H = np.eye(2)

    def predict(
        self,
        speed_mps,
        heading_rad,
        dt,
    ):
        """
        Predict next position using:

        x_new = x + speed * cos(heading) * dt
        y_new = y + speed * sin(heading) * dt
        """

        dx = (
            speed_mps
            * np.cos(heading_rad)
            * dt
        )

        dy = (
            speed_mps
            * np.sin(heading_rad)
            * dt
        )

        # Update predicted state
        self.x[0, 0] += dx
        self.x[1, 0] += dy

        # State transition matrix
        F = np.eye(2)

        # Predict covariance
        self.P = (
            F @ self.P @ F.T
            + self.Q
        )

        return self.x.copy()

    def update(
        self,
        measured_x,
        measured_y,
    ):
        """
        Correct prediction using GNSS position.
        """

        z = np.array(
            [
                [measured_x],
                [measured_y],
            ],
            dtype=float,
        )

        # Innovation / measurement residual
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = (
            self.H
            @ self.P
            @ self.H.T
            + self.R
        )

        # Kalman gain
        K = (
            self.P
            @ self.H.T
            @ np.linalg.inv(S)
        )

        # Correct state
        self.x = (
            self.x
            + K @ y
        )

        # Update covariance
        I = np.eye(2)

        self.P = (
            I - K @ self.H
        ) @ self.P

        return self.x.copy()

    def get_position(self):
        """
        Return current estimated position.
        """

        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
        )