"""Genetic Algorithm."""
import codecs
import copy
import random
from itertools import cycle, islice

from contextlib import contextmanager
from datetime import datetime, timedelta
from math import ceil, floor, fabs, log

import networkx as nx
import matplotlib.pyplot as plt
import numpy
from deap import base
from deap import creator
from deap import tools


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
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Distancia de Chebyschev
    # return ((x1 - x2)**2 + (y1 - y2)**2)**(1 / 2) # Distancia Euclidiana


def midPoint(x1, y1, x2, y2):
    """."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(indiv, f):
    """."""
    individuo = decode(indiv)
    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    x1, y1, x, y = [], [], [], []
    colors = ['red', 'gray', 'yellow']
    cutA = 1
    i3 = individuo[0][0]
    a3 = edges[i3]
    if not individuo[1][0] == 0:
        a3 = edges[i3][::-1]
    if 'D' in a3:
        a3.remove('D')
        a3.append('D')
    deslocamentos = []
    x.append(a3[0][0])
    y.append(a3[0][1])
    x.append(a3[1][0])
    y.append(a3[1][1])

    if 'D' in a3:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[2])
        f1_axes.annotate(str(cutA), midPoint(*a3[0], *a3[1]))
    else:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[0])
        f1_axes.annotate(str(cutA), midPoint(*a3[0], *a3[1]))

    cutA += 1
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]  # proxima aresta
        a1 = edges[i1]
        if not individuo[1][i] == 0:
            a1 = edges[i1][::-1]  

        if 'D' in a1:
            a1.remove('D')
            a1.append('D')

        a2 = edges[i2]
        if not individuo[1][i + 1 if i + 1 < len(individuo[0]) else 0] == 0:
            a2 = edges[i2][::-1]
        if 'D' in a2:
            a2.remove('D')
            a2.append('D')

        x1, y1, x, y = [], [], [], []
        isDeslocamento = a1[1] != a2[0]
        if isDeslocamento:  # if the next one doesn't start where the first one ends
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
        if 'D' in a2:
            f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
            f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[2])
        else:
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
    fig1.savefig(f'resultados/a-brkga-heuristica/{f}.png')
    plt.close()


def selectNext(x, listKK):
    select = None
    for i in listKK:
        if i[0] == x:
            index = random.randint(1, len(i[1]))
            select = i[1].pop(index - 1)

            if len(i[1]) == 0:
                listKK.remove(i)
            break

    if select == None:
        index = random.randint(1, len(listKK))
        removedNone = listKK[index - 1]
        listRemovedNone = removedNone[1]
        index = random.randint(1, len(listRemovedNone))
        select = listRemovedNone.pop(index - 1)
        x = removedNone[0]
        if listRemovedNone == []:
            listKK.remove(removedNone)


    second = None


    if x == select[0]:
        second = select[1]
    else:
        second = select[0]

    for k in listKK:
        if k[0] == second:
            k[1].remove(select)
            if len(k[1]) == 0:
                listKK.remove(k)
            break

    return select


def direction(p1, p2):
    xMax = p1[0] > p2[0]
    xMin = p1[0] < p2[0]
    xEqual = p1[0] == p2[0]
    yMax = p1[1] > p2[1]
    yMin = p1[1] < p2[1]
    yEqual = p1[1] == p2[1]

    return xMax, xMin, xEqual, yMax, yMin, yEqual


def newDirectionNext(after, next, isFirst):
    if isFirst:
        if after[0] == next[0] or after[0] == next[1]:
            return 0.75
        if after[1] == next[0] or after[1] == next[1]:
            return 0.25
    else:
        if after[1] == next[0]:
            return 0.25
        if after[1] == next[1]:
            return 0.75
        return random.random()


def genIndividuoRK(edges):
    """
    Generate Individuo.

    args:
        edges -> edges to cut of grapth

    individuo[0]: order of edges
    individuo[1]: order of cut

    """
    individuos = []

    size = len(listImpar)
    if size != 2:
        individuos = [random.random() for i in range(len(edges))], [
            random.random() for i in range(len(edges))], random.random(), random.random()
        return individuos

    listCopy = copy.deepcopy(edges)
    listj = listImpar
    listk = listPar

    listPoints = []
    for i in listj:
        listResult = []
        listEdges = []
        listResult.append(i)
        for j in listCopy:
            if i in j:
                if 'D' in j:
                    j.remove('D')
                listEdges.append(j)
        listResult.append(listEdges)
        listPoints.append(listResult)

    for i in listk:
        listResult = []
        listEdges = []
        listResult.append(i)
        for j in listCopy:
            if i in j:
                if 'D' in j:
                    j.remove('D')
                listEdges.append(j)
        listResult.append(listEdges)
        listPoints.append(listResult)

    index = random.randint(1, len(listj))
    select = listj[index -1]
    resultSelect = []
    while listCopy != []:

        result = selectNext(select, listPoints)
        if select == result[0]:
            select = result[1]
        else:
            select = result[0]
        resultSelect.append(result)
        listCopy.remove(result)

    p = []
    for _ in range(len(resultSelect)):
        numero = random.random()
        p.append(numero)
    p.sort(reverse=True)

    listIndividuos = [0.0] * len(resultSelect)

    for edgekk in resultSelect:
        if edgekk in edges:
            indexEdge = edges.index(edgekk)
            listIndividuos[indexEdge] = p.pop(0)
        else:
            edgekk.append('D')
            indexEdge = edges.index(edgekk)
            listIndividuos[indexEdge] = p.pop(0)

    listDirection = [0.0] * len(resultSelect)
    listResultDirection = []

    for i in range(len(resultSelect)):
        edgekk = resultSelect[i]
        if i == 0:
            result = newDirectionNext(edgekk, resultSelect[i + 1], True)
            if result < 0.5:
               listResultDirection.append(edgekk)
            else:
                listResultDirection.append([edgekk[1], edgekk[0]])
            listDirection[edges.index(edgekk)] = result
        else:
            nextIndex = i - 1
            nextEdgekk = listResultDirection[nextIndex]
            if edgekk in edges:
                result = newDirectionNext(nextEdgekk, edgekk, False)
                if result < 0.5:
                    listResultDirection.append(edgekk)
                else:
                    listResultDirection.append([edgekk[1], edgekk[0]])
                listDirection[edges.index(edgekk)] = result
            else:
                edgekk.append('D')
                result = newDirectionNext(nextEdgekk, edgekk, False)
                if result < 0.5:
                    listResultDirection.append(edgekk)
                else:
                    listResultDirection.append([edgekk[1], edgekk[0]])
                listDirection[edges.index(edgekk)] = result
    individuos.append(listIndividuos)
    individuos.append(listDirection)
    individuos.append(random.random())
    individuos.append(random.random())
    return individuos


def decode(ind):
    """."""
    listIndOrd = []
    listDirOrd = []
    indCopy = copy.deepcopy(ind[0])

    while indCopy != []:
        indexInd = 0
        maximum = indCopy[0]
        for i in range(len(indCopy)):
            if maximum < indCopy[i]:
                indexInd = i
                maximum = indCopy[indexInd]

        resultInd = indCopy.pop(indexInd)
        index = ind[0].index(resultInd)
        listIndOrd.append(index)
        listDirOrd.append(0 if ind[1][index] < 0.5 else 1)

    return [listIndOrd, listDirOrd]


def evalCut(individuo, pi=100 / 6, mi=400, decodify=True):
    """
    Eval Edges Cut.

    args:
        pi -> cutting speed
        mi -> travel speed

    if individuo[1][i] == 0 the cut is in edge order
    else the cut is in reverse edge order

    """
    ind = decode(individuo) if decodify else individuo

    dist = 0
    i3 = ind[0][0]

    a3 = edges[i3]
    if not ind[1][0] == 0:
        a3 = edges[i3][::-1]

    if 'D' in a3:
        a3.remove('D')
        a3.append('D')

    if a3[0] != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a3[0]) / mi

    if len(a3) == 3:
        dist += dist2pt(*a3[0], *a3[1]) / mi
    else:
        dist += (dist2pt(*a3[0], *a3[1])) / pi

    for i in range(len(ind[0]) - 1):
        i1 = ind[0][i]
        i2 = ind[0][i + 1]

        a1 = edges[i1]
        if not ind[1][i] == 0:
            a1 = edges[i1][::-1]
            if 'D' in a1:
                a1.remove('D')
                a1.append('D')

        a2 = edges[i2]

        if not ind[1][i + 1] == 0:
            a2 = edges[i2][::-1]
            if 'D' in a2:
                a2.remove('D')
                a2.append('D')
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        elif 'D' in a2:
            dist += (dist2pt(*a2[0], *a2[1])) / mi
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (
                dist2pt(*a2[0], *a2[1])) / pi

    iu = ind[0][-1]
    au = edges[iu]
    if not ind[1][-1] == 0:
      au = edges[iu][::-1]

    if 'D' in au:
        au.remove('D')
        au.append('D')

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
    if len(eulerian_circuit) > 0:
        edge_prox_00 = None, None, None
        for i,j in enumerate(eulerian_circuit):
            dist = dist2pt(0,0, *j[0])
            if edge_prox_00 == (None, None, None) or dist < edge_prox_00[1]:
                edge_prox_00 = (j, dist, i)
        
        path = cycle(eulerian_circuit)
        path = list(islice(path, edge_prox_00[2], edge_prox_00[2]+len(edges)))

        ind = creator.Individual([[], []])
        for i in path:
            if list(i) in edges:
                ind[0].append(edges.index(list(i)))
            elif list(i[::-1]) in edges:
                ind[0].append(edges.index(list(i[::-1])))
            ind[1].append(0 if list(i) in edges else 1)
        evalCut(ind, decodify=False)
        return [[ind], None, [ind], [0], [ind.fitness.values[0]], False]

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
        # 2
        offspring = sorted(
            list(toolbox.map(toolbox.clone, pop)),
            key=lambda x: x.fitness,
            reverse=True
        )
        # 1
        MaxGen = gamma**(log(offspring[0].fitness.values[0], gamma) - offspring[-1].fitness.values[0])
        Ke = 0.1 + ((gen / MaxGen) * (0.25 - 0.1))
        Km = 0.05 + ((1 - (gen / MaxGen)) * (0.2 - 0.05))
        alpha = 0.1 + ((1 - (gen / MaxGen)) * (0.3 - 0.1))

        # 3
        fe = offspring[0].fitness.values[0] + alpha * (offspring[-1].fitness.values[0] - offspring[0].fitness.values[0])

        tamElite = ceil(P * Ke)

        rcl = list(filter(lambda x: x.fitness.values[0] <= fe, offspring))[:tamElite]
        non_rcl = list(filter(lambda x: x.fitness.values[0] > fe, offspring))
        rcl = perturbarSimilares(rcl)

        tamMutant = ceil(P * Km if P >= Pmin and len(non_rcl) > 0 else Pmax - tamElite)

        c = []
        # Reset if population less than Pmin
        if P >= Pmin and len(non_rcl) > 0:
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


def verifyEdges(vetice, listVertices, listEdger):

    x = None
    y = None

    res = None

    quantifyMax = 100
    quantify = 0

    for index in range(len(listVertices) - 1):
        vertex = None
        if vetice[0] != listVertices[index][0] and vetice[1] != listVertices[index][1]:
            vertex = [(vetice[0], vetice[1]), (listVertices[index][0], listVertices[index][1])]
            if vertex not in listEdger:
                result = dist2pt(vetice[0], listVertices[index][0], vetice[1], listVertices[index][1])
                if res is None:
                    res = result
                    x = listVertices[index][0]
                    y = listVertices[index][1]
                else:
                    if result < res:
                        quantify = 0
                        x = listVertices[index][0]
                        y = listVertices[index][1]
                    else:
                        quantify = quantify + 1
                if quantify > quantifyMax:
                    break

    return x, y


def removeOddVerteces(vertex, listImpar, listPar, listEdger):
    x = None
    y = None
    isOdd = True

    x, y = verifyEdges(vertex, listImpar, listEdger)

    if x is None and y is None:
        x, y = verifyEdges(vertex, listPar, listEdger)
        isOdd = False

    return x, y, isOdd


def preProcess(file):
    listEdger = []

    if file:
        length = int(file[0])
        for line in range(length):
            indexLine = line + 1
            vertices = [float(vertex) for vertex in file[indexLine].split()]
            if (vertices[0], vertices[1]) in listImpar and (vertices[0], vertices[1]) not in listPar:
                listImpar.remove((vertices[0], vertices[1]))
                listPar.append((vertices[0], vertices[1]))
            elif (vertices[0], vertices[1]) in listPar and (vertices[0], vertices[1]) not in listImpar:
                listPar.remove((vertices[0], vertices[1]))
                listImpar.append((vertices[0], vertices[1]))
            else:
                listImpar.append((vertices[0], vertices[1]))

            if(vertices[2], vertices[3]) in listImpar and (vertices[2], vertices[3]) not in listPar:
                listImpar.remove((vertices[2], vertices[3]))
                listPar.append((vertices[2], vertices[3]))
            elif(vertices[2], vertices[3]) in listPar and (vertices[2], vertices[3]) not in listImpar:
                listPar.remove((vertices[2], vertices[3]))
                listImpar.append((vertices[2], vertices[3]))
            else:
                listImpar.append((vertices[2], vertices[3]))

            listEdger.append([(vertices[0], vertices[1]), (vertices[2], vertices[3])])

    isPreProcess = random.random()
    if isPreProcess < 0.5:
        return listEdger

    while listImpar:
        if len(listImpar) == 2 or len(listImpar) == 0:
            break
        if len(listImpar) == 1:
            vertex = listImpar.pop(0)
            listEdger.append([vertex, vertex, 'D'])

        else:
            index = random.randint(0, len(listImpar) - 1)
            vertex = listImpar.pop(index)
            x, y, isOdd = removeOddVerteces(vertex, listImpar, listPar, listEdger)
            listPar.append(vertex)

            if isOdd:
                listImpar.remove((x, y))
                listPar.append((x, y))
            else:
                listImpar.append((x, y))
                listPar.remove((x, y))

            if x is None and y is None:
                listEdger.append([vertex, vertex, 'D'])
            else:
                listEdger.append([vertex, (x, y), 'D'])


    return listEdger

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
            file = open(f"ejor/{t}/{f}.txt").read().strip().split('\n')

            listImpar = []
            listPar = []
            edges = []
            eulerian_circuit = None
            if file:
                edges = preProcess(file)
                G = nx.Graph()
                for i in edges:
                    G.add_edge(*(i[:2]))
                    G.add_edge(*(i[:2][::-1]))
                try:
                    eulerian_circuit = list(nx.eulerian_circuit(G))
                except:
                    eulerian_circuit = None
            # Generate Individual
            toolbox.register("indices", genIndividuoRK, edges)
            # initializ individual
            toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                toolbox.indices
            )
            # plotar(individuo, 'iteracao')
            # exit(0)
            # Generate Population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Objective Function
            toolbox.register("evaluate", evalCut)
            # function to execute map
            toolbox.register("map", map)

            hof = None
            for k in y:
                qtd = 10
                with codecs.open(f"resultados/a-brkga-heuristica/{t}/{f}_[{k}].txt", 'w+', 'utf-8') as \
                         file_write:
                    print("A-BRKGA-Heuristic:", file=file_write)
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
                        if len(iteracao) > 5:
                            print("Individual:", iteracao[2][0], file=file_write)
                        else:
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
                            f'resultados/a-brkga-heuristica/{t}/melhora/' + f"{f}_[{k}]_" +
                            str(i + 1) + '.png',
                            dpi=300
                        )
                        plt.close()
