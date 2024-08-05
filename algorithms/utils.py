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

    # Para motivos de otimização do caminho de corte,
    # podemos retirar as duplicadas de edges que são iguais e só tem sentidos diferentes
    return [*set(_edges)]


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


def save_edges_plot(edges: List[Edge], path: List[int], filename: str, duration: float):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(12, 12))

    edge_count = 0

    def plot_line(p1: Coordinate, p2: Coordinate, line_count: int, color: str = "red"):
        xs = [p1.x, p2.x]
        ys = [p1.y, p2.y]

        plt.quiver(xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0],
                   scale_units='xy', angles='xy', scale=1,
                   width=0.003,
                   color=color)

        xmid = (p1.x + p2.x) / 2
        ymid = (p1.y + p2.y) / 2

        dx = p2.x - p1.x
        dy = p2.y - p1.y
        slope = dy / dx if dx != 0 else float('inf')

        def determine_direction(x_values, y_values):
            if x_values[1] > x_values[0]:
                return 3
            elif x_values[1] < x_values[0]:
                return -3
            elif y_values[1] > y_values[0]:
                return -3
            else:
                return 3

        if abs(slope) < 1:
            ymid += determine_direction(xs, ys)
        else:
            xmid += determine_direction(xs, ys)

        plt.text(xmid, ymid, str(line_count), color='b', fontsize=10, ha='center', va='center')

    next_edge = None
    for i in range(len(path) - 1):
        edge = edges[path[i]]
        next_edge = edges[path[i + 1]]

        plot_line(p1=edge.p1, p2=edge.p2, line_count=edge_count)
        edge_count += 1

        if edge.p2 != next_edge.p1 and edge.p2 != next_edge.p2:
            e, e2 = Edge(p1=edge.p2, p2=next_edge.p1), Edge(p1=edge.p2, p2=next_edge.p2)

            if e2.distance < e.distance:
                e.reverse()

            plot_line(p1=e.p1, p2=e.p2, line_count=edge_count, color='blue')
            edge_count += 1

    if next_edge:
        plot_line(p1=next_edge.p1, p2=next_edge.p2, line_count=edge_count)

    plt.scatter(edges[path[0]].p1.x, edges[path[0]].p1.y, marker='*', s=100, color='black', zorder=10)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Optimized Edges Path: {filename}, duration {duration} seconds')
    plt.grid(True)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def save_plot_history(history: List[float], filename: str, duration: float):
    plt.clf()  # Clear the current figure
    plt.cla()  # Clear the current axes
    plt.figure(figsize=(12, 12))

    plt.plot(range(len(history)), history, 'b')  # 'bo-' means blue color, round points, solid lines

    plt.xlabel('Iteração')
    plt.ylabel('Distância percorrida')
    plt.title(f'Processo evolutivo: {filename}, duration {duration} seconds')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
