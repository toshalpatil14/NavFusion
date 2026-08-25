import math

from config import EARTH_RADIUS_M


def latlon_to_xy(
    lat_deg: float,
    lon_deg: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[float, float]:
    """
    Convert latitude/longitude to local coordinates in metres.

    x = East direction
    y = North direction
    """

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    origin_lat = math.radians(origin_lat_deg)
    origin_lon = math.radians(origin_lon_deg)

    delta_lat = lat - origin_lat
    delta_lon = lon - origin_lon

    x_m = EARTH_RADIUS_M * delta_lon * math.cos(origin_lat)
    y_m = EARTH_RADIUS_M * delta_lat

    return x_m, y_m


def xy_to_latlon(
    x_m: float,
    y_m: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[float, float]:
    """
    Convert local X/Y coordinates in metres
    back to latitude/longitude.
    """

    origin_lat = math.radians(origin_lat_deg)
    origin_lon = math.radians(origin_lon_deg)

    lat = (y_m / EARTH_RADIUS_M) + origin_lat

    lon = (
        x_m
        / (EARTH_RADIUS_M * math.cos(origin_lat))
    ) + origin_lon

    return math.degrees(lat), math.degrees(lon)