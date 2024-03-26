"""Methods for sampling of edges."""
from abc import abstractmethod
from Graph.Edge import Edge
import random
import itertools


def do_fast_nondominated_sorting(points: list[list[float]]) -> tuple[list[int], list[int]]:
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

    Args:
        points (list[list[float]]): A list of points.
    Returns:
        tuple(list[int], list[int]): Returns the ranks (non-domination fronts
        and domination counter).
    """
    layers: list[list[int]] = [[]]

    n = len(points)

    # stores the number of points by which a point is dominated
    dom_counter = [0] * n

    # non-domination rank
    ranks = [0] * n

    # list of point IDs a point dominates
    dom_elements: list[list[int]] = [[] for _ in range(n)]

    def dominates(x, y):
        n = len(x)
        assert len(x) == len(y)
        return all([x[i] <= y[i] for i in range(n)]) and any([x[i] < y[i] for i in range(n)])

    # iterate pairs of points and check dominance relation
    for i in range(n):
        for j in range(n):
            if dominates(points[i], points[j]):
                dom_elements[i].append(j)
            elif dominates(points[j], points[i]):
                dom_counter[i] += 1

        # all non-dominated points are assigned rank 1
        if dom_counter[i] == 0:
            ranks[i] = 1
            layers[0].append(i)

    # make a copy of the dominations number since we are going to modify these
    # in the next lines, but also want to return them
    dom_counter2 = dom_counter[:]

    # now determine the remaining ranks
    k = 0
    while len(layers[k]) > 0:
        layer2 = []
        for i in layers[k]:
            for j in dom_elements[i]:
                dom_counter[j] = dom_counter[j] - 1
                if dom_counter[j] == 0:
                    ranks[j] = k + 2
                    layer2.append(j)

        k += 1
        layers.append(layer2)

    return ranks, dom_counter2


class EdgeSampler(object):
    """
    Abstract EdgeSampler class.

    Expects a list of points and offers a method 'sample' to select k > 1 random
    edges. The probability distribution must be implemented in the constructor
    of realisations of this class.
    """

    @abstractmethod
    def __init__(self, edge_list: list[Edge]) -> None:
        """Initialise an EdgeSampler object."""
        # copy the edge list
        self.edge_list = edge_list[:]
        self.m = len(edge_list)
        self.cum_weights: list[float] | None = None

    def sample(self, k: int) -> list[Edge]:
        """
        Sample k > 1 edges with replacement.

        Args:
            k (int): the number of edges.
        Returns:
            list(Edge): list of Edge objects.
        """
        # random.choices() needs O(m) time if weights are passed since internally the
        # cummulated weights are calculated each time. Thus, we precalculate the
        # cummulated weights now.
        return random.choices(self.edge_list, cum_weights = self.cum_weights, k = k)


class EdgeSamplerUniform(EdgeSampler):
    """
    Random uniform edge sampling.

    I.e., each edge is sampled with probability 1/m where m is the number of edges.
    """

    def __init__(self, edge_list: list[Edge]) -> None:
        """Initialise an EdgeSamplerUniform object."""
        super().__init__(edge_list)
        # set cummulated weights
        self.cum_weights = list(itertools.accumulate([1.0 for _ in range(self.m)]))


class EdgeSamplerDominationCount(EdgeSampler):
    """
    Edge sampling biased towards the selection of edges that are dominated by few other edges.

    The probability of an edge e to be selected is proportional to (max(dc) - dc(e) + 1) / sum(dc).

    J. Bossek, C. Grimme, and F. Neumann, ‘On the benefits of biased edge-exchange
    mutation for the multi-criteria spanning tree problem’, in Proceedings of the Genetic
    and Evolutionary Computation Conference, New York, NY, USA, Jul. 2019, pp. 516–523.
    doi: 10.1145/3321707.3321818.
    """

    def __init__(self, edge_list: list[Edge]) -> None:
        """Initialise an EdgeSamplerUniform object."""
        super().__init__(edge_list)

        # calculate non-domination layers
        edge_weights = [e.get_cost() for e in edge_list]
        _, dom_counter = do_fast_nondominated_sorting(edge_weights)

        # set cummulated weights
        weights = [(max(dom_counter) - dom_counter[i]) + 1 for i in range(self.m)]
        self.cum_weights = list(itertools.accumulate(weights))
