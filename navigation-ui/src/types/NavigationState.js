export const defaultNavigationState = {
  timestamp: 0,

  latitude: 0,
  longitude: 0,

  speed: 0,
  heading: 0,

  aiSpeed: 0,
  aiConfidence: 0,

  gnssAvailable: true,

  navigationMode: "GNSS + IMU + AI + EKF",

  timeSinceGnssLoss: 0,

  positionConfidence: 0,
};