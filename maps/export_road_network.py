import osmnx as ox
import os

# --------------------------------------------------
# ROAD NETWORK EXPORT
# --------------------------------------------------

latitude = 18.5308
longitude = 73.8475

# 1. Download the same driving road network
print("Downloading road network from OpenStreetMap...")

graph = ox.graph_from_point(
    (latitude, longitude),
    dist=1500,
    network_type="drive"
)

print("Download complete!")
print("Road nodes:", len(graph.nodes))
print("Road edges:", len(graph.edges))

# --------------------------------------------------
# 2. Convert graph to GeoDataFrames
# --------------------------------------------------

nodes, edges = ox.graph_to_gdfs(graph)

print("Nodes extracted:", len(nodes))
print("Road geometries extracted:", len(edges))

# --------------------------------------------------
# 3. Create output directory
# --------------------------------------------------

output_dir = "data/road_network"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# 4. Save complete graph as GeoPackage
# --------------------------------------------------

gpkg_path = os.path.join(
    output_dir,
    "shivajinagar_road_network.gpkg"
)

ox.io.save_graph_geopackage(
    graph,
    filepath=gpkg_path
)

print("GeoPackage saved to:")
print(gpkg_path)

# --------------------------------------------------
# 5. Save road edges as GeoJSON
# --------------------------------------------------

geojson_path = os.path.join(
    output_dir,
    "shivajinagar_road_network.geojson"
)

edges.to_file(
    geojson_path,
    driver="GeoJSON"
)

print("GeoJSON saved to:")
print(geojson_path)

# --------------------------------------------------
# 6. Show useful road attributes
# --------------------------------------------------

print("\nRoad network attributes:")

useful_columns = [
    column
    for column in [
        "osmid",
        "name",
        "highway",
        "oneway",
        "maxspeed",
        "length",
        "geometry"
    ]
    if column in edges.columns
]

print(useful_columns)

print("\n====================================")
print("ROAD NETWORK EXPORT COMPLETE")
print("====================================")