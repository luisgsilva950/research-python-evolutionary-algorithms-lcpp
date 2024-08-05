import math
import random
import time
from typing import List, Tuple

from algorithms import results_repository
from algorithms.models.coordinate import Coordinate
from algorithms.models.edge import Edge
from algorithms.utils import get_edges, save_plot_history, save_edges_plot

RESULTS_DB_NAME = 'simulated_annealing_v4.sqlite'

TYPES = ['separated', 'packing']

FILES = ['instance_01_2pol', 'instance_01_3pol', 'spfc_instance']

random.seed(43)


def total_distance(path: List[int], edges: List[Edge]) -> float:
    """Calcula a distância total de um caminho percorrendo as arestas."""
    dist = 0
    for i in range(len(path) - 1):
        edge, next_edge = edges[path[i]], edges[path[i + 1]]

        if edge.p2 == next_edge.p2:
            next_edge.reverse()

        if edge.p2 == next_edge.p1:
            dist += edge.distance
            continue

        d1, d2 = distance(edge.p2, next_edge.p1), distance(edge.p2, next_edge.p2)

        if d2 < d1:
            d1 = d2

        dist += d1 + edge.distance
    return dist


DISTANCES = {}


def __distance(x1: float, x2: float, y1: float, y2: float) -> float:
    if DISTANCES.get((x1, x2, y1, y2)) is None:
        DISTANCES[(x1, x2, y1, y2)] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return DISTANCES[(x1, x2, y1, y2)]


def distance(p1: Coordinate, p2: Coordinate) -> float:
    """Calcula a distância euclidiana entre dois pontos."""
    return __distance(x1=p1.x, x2=p2.x, y1=p1.y, y2=p2.y)


def disturbance_multi_swap(path: List[int], num_swaps=1, **kwargs) -> List[int]:
    """Perturba o caminho trocando múltiplas arestas aleatoriamente."""
    # print("Executando perturbação trocando multiplas arestas")
    new_path = path[:]
    for _ in range(num_swaps):
        i, j = random.sample(range(len(path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def disturbance_shift(path: List[int], **kwargs) -> List[int]:
    """Perturba o caminho deslocando um segmento aleatório para outra posição."""
    # print("Executando perturbação deslocando segmento aleatório")
    new_path = path[:]
    i, j, k = sorted(random.sample(range(len(path)), 3))
    segment = new_path[i:j + 1]
    del new_path[i:j + 1]
    new_path[k:k] = segment
    return new_path


def disturbance_2opt(path: List[int], **kwargs) -> List[int]:
    """Perturba o caminho invertendo um segmento aleatório."""
    # print("Executando perturbação invertendo segmento aleatório")
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    if i > j:
        i, j = j, i  # Garante que i < j
    new_path[i:j + 1] = reversed(new_path[i:j + 1])
    return new_path


disturbance_reverse = lambda x, **kwargs: [*reversed(x)]


def disturbance_temperature_based(path: List[int],
                                  temperature: float,
                                  initial_temperature: float,
                                  final_temperature: float) -> List[int]:
    """Perturba o caminho com intensidade variável de acordo com a temperatura."""
    normalized_temp = (temperature - final_temperature) / (initial_temperature - final_temperature)
    perturbation_intensity = 1 - normalized_temp  # Maior no início, menor no final

    # if perturbation_intensity > 0.9:
    #     return disturbance_2opt(path)
    if perturbation_intensity > 0.8:
        return disturbance_multi_swap(path, num_swaps=5)
    elif perturbation_intensity > 0.6:
        return disturbance_multi_swap(path, num_swaps=2)
    elif perturbation_intensity > 0.3:
        fn = random.choice([disturbance_shift, disturbance_2opt, disturbance_multi_swap])
        return fn(path)
    else:
        return disturbance_multi_swap(path, num_swaps=1)


DISTURBANCES = {
    0: disturbance_shift,
    1: disturbance_2opt,
    2: disturbance_multi_swap,
    3: disturbance_temperature_based,
    4: disturbance_reverse
}


def disturbance(path: List[int], temperature: float, initial_temperature: float, final_temperature: float) -> List[int]:
    return DISTURBANCES[random.randint(0, 4)](path,
                                              temperature=temperature,
                                              initial_temperature=initial_temperature,
                                              final_temperature=final_temperature)


def simulated_annealing(edges: List[Edge],
                        t0: float = 1_000_000,
                        alpha: float = 0.999,
                        tf: float = 0.0000001,
                        internal_iterations: int = 600,
                        max_iterations: int = 100_000_000) -> Tuple[List[int], float, List[float]]:
    history = []
    path = list(range(len(edges)))
    random.shuffle(path)

    best_path = path
    best_distance = total_distance(best_path, edges)

    iteration = 0
    initial_t0 = t0

    while t0 > tf and iteration < max_iterations:
        s0_path = best_path
        s0_distance = best_distance
        for _ in range(0, internal_iterations):
            s1_path = disturbance(s0_path, temperature=t0, initial_temperature=initial_t0, final_temperature=tf)
            s1_distance = total_distance(s1_path, edges)

            delta = s1_distance - s0_distance

            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / t0):
                s0_path = s1_path
                s0_distance = s1_distance

            if s0_distance < best_distance:
                best_path = s0_path
                best_distance = s0_distance

        history.append(best_distance)

        t0 *= alpha
        iteration += 1

    return best_path, best_distance, history


if __name__ == '__main__':
    for instance_type in TYPES:
        for instance_file in FILES:

            instance = f"ejor/{instance_type}/{instance_file}.txt"
            edges = get_edges(file_name=instance)
            # plot_edges(title=f"{instance_type}/{instance_file}", edges=edges)

            for alpha in [0.7, 0.95]:
                for t0 in [1000, 10_000, 100_000]:
                    for tf in [0.00001, 0.000001]:
                        now = time.time()

                        best_path, best_distance, history = simulated_annealing(edges=edges,
                                                                                t0=t0,
                                                                                alpha=alpha,
                                                                                tf=tf,
                                                                                internal_iterations=100)

                        duration = time.time() - now

                        duration = float(int(duration * 1000)) / 1000

                        # print("Melhor percurso:", best_path, "Instance:", instance, "Params", (t0, alpha, tf))

                        print("Minor distance:", best_distance,
                              "Instance:", instance,
                              "Params", (t0, alpha, tf),
                              "Duration", duration)

                        result_image = f"results/{instance_type}/alpha_{str(alpha).replace('.', '_')}_t0_{t0}_tf_{tf}_{instance_file}.png"
                        evolution_image = f"results_evolution/{instance_type}/alpha_{str(alpha).replace('.', '_')}_t0_{t0}_tf_{tf}_{instance_file}.png"

                        result = {
                            'instance': instance,
                            'alpha': alpha,
                            't0': t0,
                            'tf': tf,
                            'duration': duration,
                            'distance': best_distance,
                            'edges_size': len(edges),
                            'result_file': result_image,
                            'evolution_image': evolution_image,
                            'n_iterations': len(history),
                            'path': str(best_path),
                            'n_internal_iterations': 100
                        }

                        results_repository.save(db=RESULTS_DB_NAME, result=result)
                        save_plot_history(history=history, filename=evolution_image, duration=duration)
                        save_edges_plot(edges=edges, path=best_path, filename=result_image, duration=duration)
