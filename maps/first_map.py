import osmnx as ox
import matplotlib.pyplot as plt

# Center point: Shivajinagar, Pune
latitude = 18.5308
longitude = 73.8475

print("Downloading road network...")

graph = ox.graph_from_point(
    (latitude, longitude),
    dist=1000,
    network_type="drive"
)

print("Download complete!")
print("Number of road nodes:", len(graph.nodes))
print("Number of road connections:", len(graph.edges))

# Plot the road network
fig, ax = ox.plot_graph(
    graph,
    node_size=0,
    edge_color="black",
    show=False,
    close=False
)

plt.show()