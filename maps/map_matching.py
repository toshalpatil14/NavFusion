import osmnx as ox
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.ops import nearest_points


# --------------------------------------------------
# 1. Reference position
# --------------------------------------------------

actual_latitude = 18.5308
actual_longitude = 73.8475

# --------------------------------------------------
# 2. Simulated noisy GPS position
# --------------------------------------------------

estimated_latitude = actual_latitude + 0.0010
estimated_longitude = actual_longitude + 0.0010

print("Actual position:")
print("Latitude:", actual_latitude)
print("Longitude:", actual_longitude)

print("\nEstimated position:")
print("Latitude:", estimated_latitude)
print("Longitude:", estimated_longitude)
# --------------------------------------------------
# Calculate position error
# --------------------------------------------------

actual_position = (
    actual_latitude,
    actual_longitude
)

estimated_position = (
    estimated_latitude,
    estimated_longitude
)

error_meters = geodesic(
    actual_position,
    estimated_position
).meters

print("\nPosition error:")
print(f"{error_meters:.2f} meters")

# --------------------------------------------------
# 3. Download road network
# --------------------------------------------------

print("\nDownloading road network...")

graph = ox.graph_from_point(
    (actual_latitude, actual_longitude),
    dist=1500,
    network_type="drive"
)

print("Road network ready!")

# --------------------------------------------------
# 4. Find nearest road to estimated position
# --------------------------------------------------

nearest_edge = ox.distance.nearest_edges(
    graph,
    X=estimated_longitude,
    Y=estimated_latitude
)

print("\nNearest road edge:")
print(nearest_edge)
# --------------------------------------------------
# 5. Find the actual closest point on that road
# --------------------------------------------------

# Convert the graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(graph)

# Get the geometry of the nearest road
u, v, key = nearest_edge

road_geometry = edges.loc[(u, v, key), "geometry"]

# Create a point for our noisy estimate
noisy_point = Point(
    estimated_longitude,
    estimated_latitude
)

# Find the closest point on the road geometry
closest_point, _ = nearest_points(
    road_geometry,
    noisy_point
)

corrected_longitude = closest_point.x
corrected_latitude = closest_point.y

print("\nMap-matched position:")
print("Latitude:", corrected_latitude)
print("Longitude:", corrected_longitude)

# Calculate remaining error after map matching
corrected_position = (
    corrected_latitude,
    corrected_longitude
)

corrected_error = geodesic(
    actual_position,
    corrected_position
).meters

print("\nError after map matching:")
print(f"{corrected_error:.2f} meters")
# --------------------------------------------------
# 5. Plot everything
# --------------------------------------------------

fig, ax = ox.plot_graph(
    graph,
    node_size=0,
    edge_color="gray",
    show=False,
    close=False
)

# Actual position
ax.scatter(
    actual_longitude,
    actual_latitude,
    s=100,
    color="green",
    zorder=5,
    label="Actual position"
)

# Noisy estimated position
ax.scatter(
    estimated_longitude,
    estimated_latitude,
    s=100,
    color="red",
    zorder=5,
    label="Noisy estimate"
)

ax.legend()

plt.show()