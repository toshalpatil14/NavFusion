"""Practical four-state planar GNSS/INS EKF."""

from __future__ import annotations

import math

import numpy as np


class PositionVelocityEKF:
    """State: [east position, north position, east velocity, north velocity]."""

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        initial_vx: float = 0.0,
        initial_vy: float = 0.0,
        process_accel_noise: float = 1.5,
        gnss_position_noise: float = 3.0,
        speed_noise: float = 1.0,
    ) -> None:
        state = (initial_x, initial_y, initial_vx, initial_vy)
        if not all(math.isfinite(value) for value in state):
            raise ValueError("Initial EKF state must be finite")
        if min(process_accel_noise, gnss_position_noise, speed_noise) <= 0:
            raise ValueError("EKF noise values must be positive")
        self.x = np.array(state, dtype=float).reshape(4, 1)
        self.P = np.diag([10.0, 10.0, 4.0, 4.0]).astype(float)
        self.process_accel_noise = float(process_accel_noise)
        self.gnss_position_noise = float(gnss_position_noise)
        self.speed_noise = float(speed_noise)

    def predict(self, ax_mps2: float, ay_mps2: float, dt_s: float) -> None:
        if not all(math.isfinite(value) for value in (ax_mps2, ay_mps2, dt_s)):
            raise ValueError("EKF prediction inputs must be finite")
        if dt_s <= 0:
            raise ValueError("EKF prediction dt must be positive")
        dt = float(dt_s)
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        control = np.array(
            [[0.5 * dt * dt * ax_mps2], [0.5 * dt * dt * ay_mps2],
             [dt * ax_mps2], [dt * ay_mps2]]
        )
        q = self.process_accel_noise ** 2
        process = q * np.array(
            [[dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
             [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
             [dt**3 / 2.0, 0.0, dt**2, 0.0],
             [0.0, dt**3 / 2.0, 0.0, dt**2]]
        )
        self.x = transition @ self.x + control
        self.P = transition @ self.P @ transition.T + process
        self._validate()

    def update_gnss(self, measured_x: float, measured_y: float) -> None:
        if not all(math.isfinite(value) for value in (measured_x, measured_y)):
            raise ValueError("GNSS measurement must be finite")
        self._update(
            np.array([[measured_x], [measured_y]]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            np.eye(2) * self.gnss_position_noise ** 2,
        )

    def update_velocity(self, vx_mps: float, vy_mps: float, noise: float | None = None) -> None:
        if not all(math.isfinite(value) for value in (vx_mps, vy_mps)):
            raise ValueError("AI velocity measurement must be finite")
        measurement_noise = self.speed_noise if noise is None else float(noise)
        if measurement_noise <= 0:
            raise ValueError("AI velocity noise must be positive")
        self._update(
            np.array([[vx_mps], [vy_mps]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.eye(2) * measurement_noise ** 2,
        )

    def update_speed(self, speed_mps: float, travel_heading_rad: float, noise: float | None = None) -> None:
        if not math.isfinite(speed_mps) or speed_mps < 0:
            raise ValueError("AI speed must be finite and non-negative")
        if not math.isfinite(travel_heading_rad):
            raise ValueError("Travel heading must be finite")
        self.update_velocity(
            speed_mps * math.sin(travel_heading_rad),
            speed_mps * math.cos(travel_heading_rad),
            noise,
        )

    def _update(self, measurement: np.ndarray, observation: np.ndarray, noise: np.ndarray) -> None:
        innovation = measurement - observation @ self.x
        covariance = observation @ self.P @ observation.T + noise
        gain = np.linalg.solve(covariance, observation @ self.P).T
        identity = np.eye(4)
        residual_transform = identity - gain @ observation
        self.x = self.x + gain @ innovation
        self.P = residual_transform @ self.P @ residual_transform.T + gain @ noise @ gain.T
        self._validate()

    def _validate(self) -> None:
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise FloatingPointError("EKF state or covariance became non-finite")
        self.P = (self.P + self.P.T) / 2.0

    def state(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.x[:, 0])

    def position(self) -> tuple[float, float]:
        return self.state()[:2]


PositionEKF = PositionVelocityEKF
