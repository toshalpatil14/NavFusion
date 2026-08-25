\# Heading \& Motion Module V1



\## Input



Smartphone IMU from IO-VNBD (10 Hz)



Sensors used:

\- Gyroscope Z

\- Orientation Azimuth

\- GPS Speed (only for motion-state labeling)



\## Output



| Field | Unit |

|--------|------|

| timestamp\_ms | ms |

| heading\_deg | degree |

| yaw\_rate | rad/s |

| motion\_state | STOP / STRAIGHT / LEFT / RIGHT |



\## Coordinate Convention



\- Heading: 0–360°

\- Positive yaw rate = Left turn

\- Negative yaw rate = Right turn



\## Sampling Rate



10 Hz

