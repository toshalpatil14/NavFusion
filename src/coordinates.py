"""Local east/north coordinate helpers."""

import math

from config import EARTH_RADIUS_M


def latlon_to_xy(lat_deg: float, lon_deg: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    origin_lat = math.radians(origin_lat_deg)
    return (
        EARTH_RADIUS_M * math.radians(lon_deg - origin_lon_deg) * math.cos(origin_lat),
        EARTH_RADIUS_M * math.radians(lat_deg - origin_lat_deg),
    )


def xy_to_latlon(x_m: float, y_m: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    origin_lat = math.radians(origin_lat_deg)
    latitude = origin_lat + y_m / EARTH_RADIUS_M
    longitude = math.radians(origin_lon_deg) + x_m / (EARTH_RADIUS_M * math.cos(origin_lat))
    return math.degrees(latitude), math.degrees(longitude)
