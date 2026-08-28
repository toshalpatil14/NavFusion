"""Typed inputs and outputs for the timestamp-based navigation EKF."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NavigationInput:
    timestamp_ms: int
    speed_mps: float
    heading_deg: float
    gnss_available: bool
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None


@dataclass(frozen=True)
class NavigationState:
    timestamp_ms: int
    dt_s: float
    x_m: float
    y_m: float
    latitude: Optional[float]
    longitude: Optional[float]
    mode: str
