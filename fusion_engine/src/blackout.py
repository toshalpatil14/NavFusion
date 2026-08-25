from config import BLACKOUT_START_SEC, BLACKOUT_END_SEC


def is_gnss_available(
    timestamp: float,
    blackout_start: float = BLACKOUT_START_SEC,
    blackout_end: float = BLACKOUT_END_SEC,
) -> bool:
    """
    Check whether GNSS is available at a given timestamp.

    GNSS is unavailable from blackout_start (inclusive)
    to blackout_end (exclusive).
    """

    return not (blackout_start <= timestamp < blackout_end)