from typing import List

from matplotlib import pyplot as plt

from algorithms.models.coordinate import Coordinate
from algorithms.models.edge import Edge


def get_edges(file_name: str) -> List[Edge]:
    with open(file_name, "r") as fp:
        file = fp.readlines()

    vertices = {}
    _edges = []
    for line in file:
        parts = line.split()
        if parts[0] == 'v':
            vertex_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            vertices[vertex_id] = Coordinate(x=x, y=y)
        elif parts[0] == 'e' and parts[3] == 'c':
            v1 = int(parts[1])
            v2 = int(parts[2])
            edge = Edge(p1=vertices[v1], p2=vertices[v2])
            _edges.append(edge)

    return _edges


def plot_edges(title: str, edges: List[Edge]):
    plt.figure(figsize=(8, 8))

    for edge in edges:
        x_values = [edge.p1.x, edge.p2.x]
        y_values = [edge.p1.y, edge.p2.y]
        plt.plot(x_values, y_values, 'bo-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"../instances/{title}.png")
    plt.close()
