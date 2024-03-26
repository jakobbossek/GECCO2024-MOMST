import math
import random
import csv
import os

def do_crowding_distance(individuals):
    m = len(individuals[0].fitness)
    n = len(individuals)

    # initialize crowding distances
    for i in range(n):
        individuals[i].cdistance = 0.0

    # determine min and max values for all objectives
    f_max = [max([ind.fitness[i] for ind in individuals]) for i in range(m)]
    f_min = [min([ind.fitness[i] for ind in individuals]) for i in range(m)]

    norm = [f_max[i] - f_min[i] for i in range(m)]
    # Workaround: sometimes we get an error because f_max[i] = f_min[i] and thus be have a zero division
    for i in range(m):
        if norm[i] < 0.001:
            norm[i] = 1

    for i in range(m):
        individuals.sort(key = lambda e: e.fitness[i])
        individuals[0].cdistance = float('inf')
        individuals[n - 1].cdistance = float('inf')
        for j in range(1, n - 1):
            individuals[j].cdistance += individuals[j].cdistance + (individuals[j + 1].fitness[i] - individuals[j - 1].fitness[i]) / norm[i]

    return individuals

def dominates(x, y):
    n = len(x)
    assert len(x) == len(y)
    return all([x[i] <= y[i] for i in range(n)]) and any([x[i] < y[i] for i in range(n)])

def do_fast_nondominated_sorting(individuals):
    """
    Fast non-dominated sorting algorithm proposed by Deb.

    Non-dominated sorting expects a set of points and returns a set of non-dominated layers. In short
    words this is done as follows: the non-dominated points of the entire set
    are determined and assigned rank 1. Afterwards all points with the current
    rank are removed, the rank is increased by one and the procedure starts again.
    This is done until the set is empty, i.e., each point is assigned a rank.

    Deb, K., Pratap, A., and Agarwal, S. A Fast and Elitist Multiobjective Genetic
    Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6 (8) (2002),
    182-197.
    """
    layers = [[]]
    layers_indices = [[]]

    n = len(individuals)

    # stores the number of individuals by which a point is dominated
    dom_counter = [0] * n

    # non-domination rank
    ranks = [0] * n

    # list of point IDs a point dominates
    dom_elements = [[] for _ in range(n)]

    # iterate pairs of individuals and check dominance relation
    for i in range(n):
        for j in range(n):
            if dominates(individuals[i].fitness, individuals[j].fitness):
                dom_elements[i].append(j)
            elif dominates(individuals[j].fitness, individuals[i].fitness):
                dom_counter[i] += 1

        # all non-dominated individuals are assigned rank 1
        if dom_counter[i] == 0:
            ranks[i] = 1
            layers[0].append(individuals[i])
            layers_indices[0].append(i)

    # now determine the remaining ranks
    k = 0
    while len(layers_indices[k]) > 0:
        layer2 = []
        layer2_indices = []
        for i in layers_indices[k]:
            for j in dom_elements[i]:
                dom_counter[j] = dom_counter[j] - 1
                if dom_counter[j] == 0:
                    ranks[j] = k + 2
                    layer2.append(individuals[j])
                    layer2_indices.append(j)

        k += 1
        layers.append(layer2)
        layers_indices.append(layer2_indices)

    return layers


class NSGA2Individual(object):

    def __init__(self, x):
        """
        NSGA-II individual object.

        Fields:
            x [any]: the individual itself (whatever encoding is used).
            fitness [list[float]]: the individuals' fitness vector.
            cdistance float: the individuals' crowding distance.
        """
        super().__init__()
        self.x = x
        self.fitness = None
        self.cdistance = 0.0

    def evaluate(self, objective_fun):
        """
        Calculate the individuals' objective function value.
        """

        if self.fitness is None:
            self.fitness = objective_fun(self.x)
        return self.fitness

    def get_solution(self):
        return self.x

    def get_fitness(self):
        return self.fitness


class NSGA2(object):

    def __init__(self,
        mu: int,
        lambdaa: int,
        initializer,
        objective_fun,
        mutator,
        max_evals: int,
        log_every: int | None = 10) -> None:
        """
        Simplified NSGA-II. This implementation deviates from the NSGA-II by Deb et al. in the following two points:
        * Parent selection is uniformly at random (i.e., not based on binary tournament selection and the partial
          crowding distance order).
        * It allows for a (mu + lambda) strategy with arbitrary integer lambda > 0.
        """
        super().__init__()

        assert mu >= 5
        assert lambdaa >= 1

        self.mu = int(mu)
        self.lambdaa = int(lambdaa)
        self.initializer = initializer
        self.objective_fun = objective_fun
        self.mutator = mutator
        self.max_evals = max_evals

        # calculate number of generations
        self.max_gens = math.ceil((self.max_evals - self.mu) / self.lambdaa)

        self.log_every = log_every

        # number of function evaluations
        self.evals = 0
        self.gens = 0

        # history
        self.pf = None
        self.history = []

    def sample_parents(self):
        return [random.randint(0, self.mu - 1) for _ in range(self.lambdaa)]

    def run(self) -> list:

        # initialise population
        P = [NSGA2Individual(self.initializer()) for _ in range(self.mu)]
        # print(f"Number of individuals in initial population: {len(P)}")

        # calculate fitness
        P_fitness = [x.evaluate(self.objective_fun) for x in P]
        self.evals = self.mu

        if self.log_every is not None:
            if (self.gens % self.log_every) == 0:
                self.history.append(P[:])

        # main loop
        while self.gens < self.max_gens:
            # generate offspring
            # print(f"|P_{self.gens}| = {len(P)}")

            parents = [P[i] for i in self.sample_parents()]
            # print(f"|parents_{self.gens}| = {len(parents)}")

            Q = [NSGA2Individual(self.mutator(x.x)) for x in parents]
            Q_fitness = [x.evaluate(self.objective_fun) for x in Q]
            self.evals += self.lambdaa
            self.gens += 1

            # Combine population and offspring
            R = P + Q
            # print(f"|R_{self.gens}| = {len(R)}")
            # print(len(R))

            # Now do the survival selection
            layers = do_fast_nondominated_sorting(R)
            #print(layers)

            P = []
            i = 0
            while ((len(P) + len(layers[i])) <= self.mu):
                P = P + layers[i] # append the entire layer
                i += 1

            # number of elements that need to be selected for to complete to next population
            if len(P) < self.mu:
                no_missing = self.mu - len(P)

                # now do crowding distance calculation on the i-th layer
                remaining = do_crowding_distance(layers[i])
                # print(f"{self.gens}: |P| = {len(P)}, missing = {no_missing}, |remaining| = {len(remaining)}")

                remaining.sort(key = lambda e: e.cdistance, reverse = True)

                P = P + remaining[0:no_missing]

            if self.log_every is not None:
                if (self.gens % self.log_every) == 0:
                    self.history.append(P[:])

        if self.log_every is not None:
            if (self.gens % self.log_every) == 0:
                self.history.append(P[:])

        self.final_population = P

        return P, self.history

    def write_result(self, outpath, total_time):
        # write trajectory/history
        file = os.path.join(outpath, "trajectory.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow(["gen","c1","c2"])
            for i, pop in enumerate(self.history):
                gen = i * self.log_every
                for ind in pop:
                    csvwriter.writerow([str(gen)] + [str(x) for x in ind.fitness])

        # final approximation
        file = os.path.join(outpath, "pf.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow(["c1","c2"])
            for ind in self.final_population:
                csvwriter.writerow([str(x) for x in ind.fitness])

        # time
        file = os.path.join(outpath, "runtime.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow([str(total_time)])

