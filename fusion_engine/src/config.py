# config.py

# Earth's average radius in metres
EARTH_RADIUS_M = 6_371_000.0

# Default navigation update rate
UPDATE_RATE_HZ = 10.0
DEFAULT_DT = 1.0 / UPDATE_RATE_HZ

# Default GNSS blackout period for testing
BLACKOUT_START_SEC = 10.0
BLACKOUT_END_SEC = 30.0

# Navigation modes
MODE_GNSS_INS = "GNSS_INS"
MODE_DEAD_RECKONING = "DEAD_RECKONING"