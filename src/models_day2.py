from dataclasses import dataclass
from typing import Optional


@dataclass
class NavigationInput:
    timestamp: float
    speed_mps: float
    heading_rad: float

    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None

    gnss_available: bool = False


@dataclass
class NavigationState:
    timestamp: float

    x_m: float
    y_m: float

    latitude: Optional[float]
    longitude: Optional[float]

    speed_mps: float
    heading_rad: float

    mode: str
    confidence: float