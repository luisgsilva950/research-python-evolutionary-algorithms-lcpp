"""Genetic Algorithm."""
import codecs
import random

import numpy

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime, timedelta

from math import ceil, floor, fabs, log


@contextmanager
def timeit(file_write=None):
    """Context Manager to check runtime."""
    start_time = datetime.now()
    print(f'Start Time (hh:mm:ss.ms) {start_time}', file=file_write)
    yield
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f'End Time (hh:mm:ss.ms) {end_time}', file=file_write)
    print(f'Total Time (hh:mm:ss.ms) {time_elapsed}', file=file_write)


def dist2pt(x1, y1, x2, y2):
    """."""
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Chebyschev Distance


def midPoint(x1, y1, x2, y2):
    """."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(indiv, f):
    """."""
    individuo = decode(indiv)
    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    x1, y1, x, y = [], [], [], []
    colors = ['red', 'gray']
    cutA = 1
    i1 = individuo[0][0]
    a1 = edges[i1] if individuo[1][0] == 0 else edges[i1][::-1]
    deslocamentos = []
    x.append(a1[0][0])
    y.append(a1[0][1])
    x.append(a1[1][0])
    y.append(a1[1][1])
    f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                   scale_units='xy', angles='xy', scale=1, color=colors[0])
    f1_axes.annotate(str(cutA), midPoint(*a1[0], *a1[1]))
    cutA += 1
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]  # proxima aresta
        a1 = edges[i1] if individuo[1][i] == 0 else edges[i1][::-1]  # atual
        a2 = edges[i2] if individuo[1][
            i + 1 if i + 1 < len(individuo[0]) else 0] == 0 else edges[i2][::-1]  # proxima
        x1, y1, x, y = [], [], [], []
        if a1[1] != a2[0]:  # if the next one doesn't start where the first one ends
            x1.append(a1[1][0])
            y1.append(a1[1][1])
            x1.append(a2[0][0])
            y1.append(a2[0][1])
            deslocamentos.append({
                'pontos': [x1[0], y1[0], x1[1] - x1[0], y1[1] - y1[0]],
                'annot': str(cutA),
                'mid': midPoint(*a1[1], *a2[0])
            })
            cutA += 1
        # plot next
        x.append(a2[0][0])
        y.append(a2[0][1])
        x.append(a2[1][0])
        y.append(a2[1][1])
        f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[0])
        cutA += 1
    for i in deslocamentos:
        f1_axes.annotate(i['annot'], (i['mid'][0] - 3, i['mid'][1]))
        f1_axes.quiver(*i['pontos'], width=.005,
                       scale_units='xy', angles='xy', scale=1, color=colors[1])
    f1_axes.set_xlim(*f1_axes.get_xlim())
    f1_axes.set_ylim(*f1_axes.get_ylim())
    plt.title("Time Required: {:.2f}".format(indiv.fitness.values[0]))
    fig1.savefig(f'resultados/a-brkga/{f}.png')
    plt.close()


def genIndividuoRK(edges):
    """
    Generate Individuo.

    args:
        edges -> edges to cut of grapth

    individuo[0]: order of edges
    individuo[1]: order of cut

    """
    return [
        random.random() for i in range(len(edges))
    ], [
        random.random() for i in range(len(edges))
    ], random.random(), random.random()


def decode(ind):
    """."""
    return [ind[0].index(i) for i in sorted(ind[0])], [0 if i < 0.5 else 1 for i in ind[1]]


def evalCut(individuo, pi=100 / 6, mi=400):
    """
    Eval Edges Cut.

    args:
        pi -> cutting speed
        mi -> travel speed

    if individuo[1][i] == 0 the cut is in edge order
    else the cut is in reverse edge order

    """
    ind = decode(individuo)
    dist = 0
    i1 = ind[0][0]
    a1 = edges[i1] if ind[1][0] == 0 else edges[i1][::-1]
    if a1 != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a1[0]) / mi
    dist += (dist2pt(*a1[0], *a1[1])) / pi
    for i in range(len(ind[0]) - 1):
        i1 = ind[0][i]
        i2 = ind[0][i + 1 if i + 1 < len(ind[0]) else 0]
        a1 = edges[i1] if ind[1][i] == 0 else edges[i1][::-1]
        a2 = edges[i2] if ind[1][i + 1 if i + 1 < len(
            ind[0]
        ) else 0] == 0 else edges[i2][::-1]
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (
                dist2pt(*a2[0], *a2[1])) / pi
    iu = ind[0][-1]
    au = edges[iu] if ind[1][-1] == 0 else edges[iu][::-1]
    if au != (0.0, 0.0):
        dist += dist2pt(*au[1], 0.0, 0.0) / mi
    individuo.fitness.values = (dist, )
    return dist,


def calcBeta(ind):
    """."""
    return 0.001 + ind[3] * (0.1 - 0.001)


def calcPe(ind):
    """."""
    return 0.65 + ind[2] * (0.80 - 0.65)


def perturbarSimilares(pop):
    """."""
    def addItem(x):
        if x.fitness in fits.keys():
            fits[x.fitness].append(x)
        else:
            fits[x.fitness] = [x]

    def pertuba(x):
        beta = calcBeta(x)
        for i in range(len(x[0])):
            x[0][i] = x[0][i] if random.random() > beta else random.random()
        for i in range(len(x[1])):
            x[1][i] = x[1][i] if random.random() > beta else random.random()
        x[2] = x[2] if random.random() > beta else random.random()
        x[3] = x[3] if random.random() > beta else random.random()

    fits = {}
    list(
        map(
            addItem,
            pop
        )
    )
    list(
        map(
            lambda x: map(pertuba, x[1]),
            filter(
                lambda x: len(x[1]) > 1,
                fits.items()
            )
        )
    )
    pop = []
    [pop.extend(i) for i in fits.values()]
    return pop


def main(Pmax=1000, Pmin=100, gamma=0.999, file=None):
    """
    Execute Genetic Algorithm.

    args:
        P -> size of population
        Pe -> size of elite population
        Pm -> size of mutant population
        Pe -> elite allele inheritance probability
        MaxGen -> Number of generations without converge
        file -> if write results in file

    """
    pop = toolbox.population(n=Pmax)

    toolbox.register("mate", crossBRKGA)

    gen, genMelhor = 0, 0

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Evaluate the entire population
    list(toolbox.map(toolbox.evaluate, pop))
    melhor = numpy.min([i.fitness.values for i in pop])
    logbook = tools.Logbook()
    p = stats.compile(pop)
    logbook.record(
        generations=0,
        MaxGenenerations=0,
        Populationk1=0,
        Populationk=0,
        PopulationElite=0,
        PopulationMutant=0,
        fitnessElite=0,
        alpha=0, **p)
    logbook.header = "generations", 'MaxGenenerations', 'Populationk1', 'Populationk', 'PopulationElite', 'PopulationMutant', 'fitnessElite', 'alpha', 'min', 'max', "avg", "std"
    gens, inds = [], []
    gens.append(gen)
    inds.append(melhor)
    print(logbook.stream, file=file)
    MaxGen = 100
    while gen <= MaxGen:
        # Select the next generation individuals
        Pk = len(pop)
        P = floor(len(pop) * gamma)

        offspring = sorted(
            list(toolbox.map(toolbox.clone, pop)),
            key=lambda x: x.fitness,
            reverse=True
        )
        
        MaxGen = gamma**(log(offspring[0].fitness.values[0], gamma) - offspring[-1].fitness.values[0])
        Ke = 0.1 + ((gen / MaxGen) * (0.25 - 0.1))
        Km = 0.05 + ((1 - (gen / MaxGen)) * (0.2 - 0.05))
        alpha = 0.1 + ((1 - (gen / MaxGen)) * (0.3 - 0.1))

        fe = offspring[0].fitness.values[0] + alpha * (offspring[-1].fitness.values[0] - offspring[0].fitness.values[0])

        tamElite = ceil(P * Ke)
        tamMutant = ceil(P * Km if P >= Pmin else Pmax - tamElite)

        rcl = list(filter(lambda x: x.fitness.values[0] <= fe, offspring))[:tamElite]
        non_rcl = list(filter(lambda x: x.fitness.values[0] > fe, offspring))
        rcl = perturbarSimilares(rcl)

        c = []
        # Reset if population less than Pmin
        if P >= Pmin:
            tamCrossover = P - len(rcl) - tamMutant

            # Apply crossover and mutation on the offspring
            for _ in range(tamCrossover):
                a = random.choice(rcl)
                b = random.choice(non_rcl)
                ni = creator.Individual([*toolbox.mate(a, b)])
                c.append(ni)

        Pm = toolbox.population(n=tamMutant)

        c = rcl + c + Pm
        offspring = c

        list(toolbox.map(toolbox.evaluate, offspring[len(rcl):]))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        gen += 1
        minf = numpy.min([i.fitness.values for i in pop])
        men = False
        try:
            if minf < melhor:
                men = True
                melhor = minf
                genMelhor = gen
        except Exception:
            print(minf)

        p = stats.compile(pop)
        logbook.record(
            generations=gen,
            MaxGenenerations=MaxGen,
            Populationk1=P,
            Populationk=Pk,
            PopulationElite=tamElite,
            PopulationMutant=tamMutant,
            fitnessElite=fe,
            alpha=str(alpha)[:5],
            **p
        )
        if gen - genMelhor <= MaxGen and not men:
            print(logbook.stream)
        else:
            print(logbook.stream, file=file)
        hof.update(pop)
        gens.append(gen)
        inds.append(minf)
    return pop, stats, hof, gens, inds


def crossBRKGA(a, b):
    """."""
    return [
        a[0][i] if random.random() < calcPe(b) else b[0][i]
        for i in range(min(len(a[0]), len(b[0])))
    ], [
        a[1][i] if random.random() < calcPe(b) else b[1][i]
        for i in range(min(len(a[1]), len(b[1])))
    ], a[2] if random.random() < calcPe(b) else b[2], a[3] if random.random() < calcPe(b) else b[3]


files = [
    'instance_01_2pol',
    'instance_01_3pol',
    'instance_01_4pol',
    'instance_01_5pol',
    'instance_01_6pol',
    'instance_01_7pol',
    'instance_01_8pol',
    'instance_01_9pol',
    'instance_01_10pol',
    'instance_01_16pol',
    'albano',
    'blaz1',
    'blaz2',
    'blaz3',
    'dighe1',
    'dighe2',
    'fu',
    'rco1',
    'rco2',
    'rco3',
    'shapes2',
    'shapes4',
    'instance_artificial_01_26pol_hole',
    'spfc_instance',
    'trousers',
]

y = [
    .999,
    .998,
    .997,
]
# toolbox of GA
toolbox = base.Toolbox()
# Class Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Representation Individual
creator.create("Individual", list, fitness=creator.FitnessMin)
tipo = [
    'packing',
    'separated'
]
if __name__ == "__main__":
    for t in tipo:
        for f in files:
            file = open(f"ejor/{t}/{f}.txt").read(
            ).strip().split('\n')
            edges = []
            if file:
                n = int(file.pop(0))
                for i in range(len(file)):
                    a = [float(j) for j in file[i].split()]
                    edges.append([(a[0], a[1]), (a[2], a[3])])
            # Generate Individual
            toolbox.register("indices", genIndividuoRK, edges)
            # initializ individual
            toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                toolbox.indices
            )
            # Generate Population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Objective Function
            toolbox.register("evaluate", evalCut)
            # function to execute map
            toolbox.register("map", map)

            hof = None
            for k in y:
                qtd = 10
                with codecs.open(f"resultados/a-brkga/{t}/{f}_[{k}].txt", 'w+',"utf-8") as \
                         file_write:
                    print("A-BRKGA:", file=file_write)
                    print(file=file_write)
                    for i in range(qtd):
                        print(f"Execution {i+1}:", file=file_write)
                        print(
                            f"Parameters: Y={k}",
                            file=file_write
                        )
                        iteracao = None
                        with timeit(file_write=file_write):
                            iteracao = main(
                                gamma=k,
                                file=file_write
                            )
                        print("Individual:", decode(iteracao[2][0]), file=file_write)
                        print("Fitness: ", iteracao[2][0].fitness.values[0], file=file_write)
                        print("Gens: ", iteracao[3], file=file_write)
                        print("Inds: ", iteracao[4], file=file_write)
                        print(file=file_write)
                        plotar(iteracao[2][0], f"{t}/plot/{f}_[{k}]_" + str(i + 1))
                        fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
                        fig1.set_size_inches((10, 10))
                        gens, inds = iteracao[3], iteracao[4]
                        f1_axes.set_ylabel("Best Individual Value")
                        f1_axes.set_xlabel("Generations")
                        f1_axes.grid(True)
                        f1_axes.set_xlim(0, gens[-1])
                        f1_axes.set_ylim(inds[-1] - 10, inds[0] + 10)
                        f1_axes.plot(gens, inds, color='blue')
                        fig1.savefig(
                            f'resultados/a-brkga/{t}/melhora/' + f"{f}_[{k}]_" +
                            str(i + 1) + '.png',
                            dpi=300
                        )
                        plt.close()
