"""Different (pseudo-random) graph generators."""
from Graph.Graph import Graph
from Graph.Edge import Edge
from Graph.algorithms.CC import CC
import random
import math
from typing import Optional

# TODO: rounding to nearest integer (there seems to be no build-in method for this)
# TODO: eta in concave?
# TODO: rounding in concave?
# TODO: more general weights in concave?
# TODO: throw warning if asint=True, but max - min < 10?
# TODO: anti-correlated output is not what we expect.


class GraphGenerator:
    """Methods for the iterative generation of multi-weighted graphs."""

    @staticmethod
    def add_points(graph: Graph, min: float = 0, max: float = 1) -> Graph:
        """Add 2d point coordinates in [min,max] x [min,max] at random."""
        graph.points = [(random.uniform(min, max), random.uniform(min, max)) for _ in range(1, graph.get_n() + 1)]
        return graph

    @staticmethod
    def euclidean_distance(x: list[float], y: list[float]) -> float:
        """Calculate Euclidean distance between two points."""
        dists = [(x[i] - y[i]) * (x[i] - y[i]) for i in range(len(x))]
        return math.sqrt(sum(dists))

    @staticmethod
    def add_edges_complete(n: int) -> Graph:
        """Generate a complete graph with n nodes and all pairwise edges."""
        return GraphGenerator.add_edges_erdosrenyi(n, 1)

    @staticmethod
    def add_edges_star(n: int, center: int = 1) -> Graph:
        """Generate a star graph with n nodes."""
        graph = Graph(n)
        for i in range(1, n + 1):
            if i == center:
                continue
            graph.add_edge(Edge(center, i), False)
        return graph

    @staticmethod
    def add_edges_line(n: int) -> Graph:
        """Generate a line graph with n nodes."""
        graph = Graph(n)
        for v in range(1, n):
            graph.add_edge(Edge(v, v + 1), False)

        assert graph.get_m() == (n - 1)
        return graph

    @staticmethod
    def add_edges_empty(n: int) -> Graph:
        """Generate an empty graph with n nodes."""
        return GraphGenerator.add_edges_erdosrenyi(n, 0)

    @staticmethod
    def add_edges_erdosrenyi(n: int, p: float) -> Graph:
        """Generate an Erdös-Renyi graph with n nodes."""
        assert n >= 2
        assert 0 <= p <= 1

        graph = Graph(n)

        graph = GraphGenerator.add_points(graph)
        for v in range(1, n + 1):
            for w in range(v + 1, n + 1):
                if random.random() > p:
                    continue

                # TODO: 'plain' graphs should not be weighted
                graph.add_edge(Edge(v, w), False)

        return graph

    @staticmethod
    def add_weights_random(graph: Graph, q: int = 2, min = 0, max = 1, asint: bool = False) -> Graph:
        """
        Random edge weight generator.

        Simply sets each component of each edge weight vector to a value
        drawn from a uniformly random distribution U(min, max).

        Args:
            graph (Graph): source graph to be augmented with additional edge weights.
            q (int): number of edge weights to add. Default is 2.
            min (int): minimum weight; defaults to 0.
            max (int): maximum weight; defaults to 1.
            asint (bool): Should weights be rounded to their nearest integer?
                Default is False.

        Returns:
            A copy of 'graph' augmeted with q additional weights per edge.
        """
        # iterate over edges and modify weights
        graph2 = graph.copy(copy_edges = False)
        graph2.q += q
        for e in graph:
            # sample weights
            v, w, costs = e.v, e.w, e.get_cost()
            new_costs = [random.uniform(min, max) for _ in range(q)]
            if asint:
                new_costs = [int(w) for w in new_costs]
            costs = costs + new_costs
            graph2.add_edge(Edge(v, w, costs), check_exists = False)

        return graph2

    @staticmethod
    def add_weights_correlated(graph: Graph, alpha: float, q: int = 2, asint: bool = False) -> Graph:
        """
        (Anti)-correlated edge weight generator.

        Generates weighs as follows: the first weight is sampled from a U(0,1)
        distribution. All other weight 2,...,q are either positively or negatively
        correlated with the first weight. The strength of the correlation can be
        adjusted via the alpha parameter. See Algorithm 1 in the paper for details:
        J. D. Knowles and D. W. Corne, ‘Benchmark Problem Generators and Results
        for the Multiobjective Degree-Constrained Minimum Spanning Tree Problem’,
        p. 8.

        Args:
            graph (Graph): source graph to be augmented with additional edge weights.
            alpha (float): required correlation (value between -1 and 1).
            q (int): number of edge weights to add. Default is 2.
            asint (bool): Should weights be rounded to their nearest integer?
                Default is False.

        Returns:
            A copy of 'graph' augmeted with q additional weights per edge.
        """
        assert -1 <= alpha <= 1

        if alpha >= 0:
            beta = 0.5 * (1 - alpha)
            gamma = beta
        else:
            beta = 0.5 * (1 + alpha)
            gamma = beta - alpha

        graph2 = graph.copy(copy_edges = False)
        graph2.q += q
        for e in graph:
            # sample first weight
            new_costs = [0] * q
            new_costs[0] = random.random()
            for i in range(1, q):
                new_costs[i] = alpha * new_costs[0] + beta + gamma * random.uniform(-1, 1)

            if asint:
                new_costs = [int(w) for w in new_costs]

            v, w, costs = e.v, e.w, e.get_cost()
            costs = costs + new_costs
            graph2.add_edge(Edge(v, w, costs), check_exists = False)

        return graph2

    @staticmethod
    def add_weights_concave(
        graph: Graph,
        q: int = 2,
        zeta: Optional[float] = None,
        eta: float = 0.2,
        asint: bool = False
    ) -> Graph:
        """
        Concave edge weight generator.

        Generates weighs as follows: without loss of generality the first three nodes
        serve as 'special' nodes. Find more details in Section 3 of the respective paper:
        J. D. Knowles and D. W. Corne, ‘Benchmark Problem Generators and Results
        for the Multiobjective Degree-Constrained Minimum Spanning Tree Problem’,
        p. 8.

        Args:
            graph (Graph): source graph to be augmented with additional edge weights.
            q (int): number of edge weights to add. Default is 2 (other values are not
                accepted so far).
            zeta (float): parameter zeta. Default is 1/n where n is the number of
                nodes of the source graph.
            eta (float): parameter eta. Default is 0.2.
            asint (bool): Should weights be rounded to their nearest integer?
                Default is False.

        Returns:
            A copy of 'graph' augmeted with q additional weights per edge.
        """
        if q != 2:
            raise ValueError('q <> 2 is currently not supported!')

        graph2 = graph.copy(copy_edges = False)
        graph2.q += q

        if zeta is None:
            # proportional to 1/n (see paper)
            zeta = 1 / graph.get_n()

        # W.l.o.g. the first three nodes {1,2,3} are the 'special' vertices
        w12 = [zeta, zeta]
        w13 = [0, 1 - zeta]
        w23 = [1 - zeta, 0]

        # add 'special' edges
        graph2.add_edge(Edge(1, 2, w12), check_exists = False)
        graph2.add_edge(Edge(1, 3, w13), check_exists = False)
        graph2.add_edge(Edge(2, 3, w23), check_exists = False)

        # now iterate over the remaining nodes
        for e in graph:
            v, w, costs = e.v, e.w, e.get_cost()
            new_costs = []

            if v > 3 and w > 3:
                new_costs = [random.uniform(zeta, eta), random.uniform(zeta, eta)]

            # ^ = xor
            elif (v <= 3) ^ (w <= 3):
                new_costs = [random.uniform(1 - zeta, 1), random.uniform(1 - zeta, 1)]

            else:
                # do not add any edges between the 'specific' nodes {1,2,3} anymore
                continue

            costs = costs + new_costs

            graph2.add_edge(Edge(v, w, costs), check_exists = False)

        return graph2

    @staticmethod
    def add_edges_m_concave(
        graph: Graph,
        f: int,
        fld: int,
        fud: int,
        alpha: float,
        chi: float = 1,
        phi: float = 0.5,
        xi: float = 100,
        omega: float = 110
    ) -> Graph:
        """
        Generate hard to solve (mc)-d-MST instances.

        Args:
            graph (Graph): source graph to be augmented with edges.
            f (int): the desired number of vertices with large degree.
            fld (int): the lower bound on the degree of large-degree vertices.
            fud (int): the upper bound on the degree of large-degree vertices.
            chi (float): upper bound for the height degree edges.
            xi (float): bound parameters.
            phi (float): lower bound of the weight of high
            alpha (float): required correlation (value between -1 and 1).

        Returns:
            A copy of 'graph'.
        """
        n = graph.get_n()
        assert fld < n and fld <= fud, 'Constraints on lower and/or upper bounds are violated.'

        # Second weight will be correlated to the first one.
        assert -1 <= alpha <= 1

        if alpha >= 0:
            beta = 0.5 * (1 - alpha)
            gamma = beta
        else:
            beta = 0.5 * (1 + alpha)
            gamma = beta - alpha

        tmp_graph = graph.copy(copy_edges = False)

        # STAGE 1) BUILD MST

        # 1a) sample f random 'star'-like graph centers
        node_ids = list(range(1, n + 1))
        centers = random.sample(node_ids, f)

        # 1b) sample random star graphs (i.e., sample each [fld, fud] neigbours and add edge)
        for v in centers:
            neighbor_ids = node_ids[:]  # expensive!
            neighbor_ids.remove(v)  # expensive!

            # sample neighborhood size
            k = random.randint(fld, fud)
            neighbors = random.sample(neighbor_ids, k)
            for w in neighbors:
                tmp_graph.add_edge(Edge(v, w, [random.uniform(0, chi)]), check_exists = True)

        # get CCs of the current status
        ccs = CC(tmp_graph).get_components()
        ccs_large = [cc for cc in ccs if len(cc) > 1]

        # ... and add up to (f-1) random edges to form a tree (not necessarily spanning tree)
        n_ccs = len(ccs_large)
        if n_ccs > 1:
            # random order of components
            order = list(range(n_ccs))
            random.shuffle(order)

            # now add random links between two consecutive components
            for i in range(n_ccs - 1):
                v = random.sample(ccs_large[order[i]], 1)[0]
                w = random.sample(ccs_large[order[i + 1]], 1)[0]
                tmp_graph.add_edge(Edge(v, w, [random.uniform(0, chi)]), check_exists = False)

        # 1c) link add all remaining (isolated) nodes to random star graph centers
        # with weight in [phi, chi]
        nodes_isolated = tmp_graph.get_isolated_nodes()
        for u in nodes_isolated:
            v = random.sample(centers, 1)[0]  # random center
            tmp_graph.add_edge(Edge(u, v, [random.uniform(phi, chi)]), check_exists = False)

        # STAGE 2) ADD REMAINING EDGES

        # 2a) Eventually, add any edges between non-adjacent nodes until
        # a desired density is reached.
        # get nodes of the 'connected part' so far (largest CC)
        ccs = CC(tmp_graph).get_components()
        nodes_ccs_largest = [cc for cc in ccs if len(cc) > 1][0]

        in_largest = [False] * (n + 1)
        for v in nodes_ccs_largest:
            in_largest[v] = True

        tmp_graph_final = tmp_graph.copy(copy_edges = True)
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if u == v:
                    continue
                # check if the edge is part of the large CC
                if tmp_graph_final.has_edge(u, v):
                    continue

                costs = [random.uniform(chi, omega)] if (in_largest[u] and in_largest[v]) else [random.uniform(phi, omega)]
                tmp_graph_final.add_edge(Edge(u, v, costs), check_exists = False)

        # eventually add the correlated weight
        final_graph = tmp_graph_final.copy(copy_edges = False)
        for e in tmp_graph_final:
            u, v, costs = e.get_v(), e.get_w(), e.get_cost()
            costs = costs.append(alpha * costs[0] + beta + gamma * random.uniform(-1, 1))
            final_graph.add_edge(Edge(u, v, costs), check_exists = False)

        return final_graph

    @staticmethod
    def add_weights_unit(graph: Graph, q: int = 1) -> Graph:
        """Add single unit weight (i.e., 1) to every edge."""
        graph2 = graph.copy(copy_edges = False)
        graph2.q += q

        for e in graph:
            v, w, costs = e.v, e.w, e.get_cost()
            new_costs = [1] * q

            graph2.add_edge(Edge(v, w, costs + new_costs), check_exists = False)

        return graph2

    @staticmethod
    def generate_kc_random(n: int, p: float = 1) -> Graph:
        """Bi-weighted random Knowles & Corne graph."""
        graph = GraphGenerator.add_edges_erdosrenyi(n, p)
        return GraphGenerator.add_weights_random(graph)

    @staticmethod
    def generate_kc_correlated(n: int, p: float = 1, alpha: float = 0.7) -> Graph:
        """Bi-weighted positively correlated Knowles & Corne graph."""
        graph = GraphGenerator.add_edges_erdosrenyi(n, p)
        return GraphGenerator.add_weights_correlated(graph, alpha = alpha)

    @staticmethod
    def generate_kc_anticorrelated(n: int, p: float = 1, alpha: float = -0.7) -> Graph:
        """Bi-weighted negatively correlated Knowles & Corne graph."""
        return GraphGenerator.generate_kc_correlated(n, alpha)

    @staticmethod
    def generate_kc_concave(n: int, p: float = 1, zeta: float = 0.03, eta: float = 0.0125) -> Graph:
        """Bi-weighted Knowles & Corne graph with concave Pareto-front."""
        graph = GraphGenerator.add_edges_erdosrenyi(n, p)
        return GraphGenerator.add_weights_concave(graph, zeta = zeta, eta = eta)

    @staticmethod
    def generate_kc_m_correlated(
        n: int,
        f: int,
        fld: int,
        fud: int,
        alpha: float,
        chi: float = 1,
        phi: float = 0.5,
        xi: float = 100,
        omega: float = 110
    ) -> Graph:
        """Bi-weighted Knowles & Corne graph which is misleading for the (mc)-d-MST."""
        graph = Graph(n)
        return GraphGenerator.add_edges_m_concave(graph, f, fld, fud, alpha, chi, phi, xi, omega)

    @staticmethod
    def generate_unit_weighted(n: int, p: float = 1) -> Graph:
        """Generate simple graph with unit-weights."""
        graph = GraphGenerator.add_edges_erdosrenyi(n, p)
        return GraphGenerator.add_weights_unit(graph)

    @staticmethod
    def generate_euclidean(n: int, p: float = 1) -> Graph:
        """Generate simple Euclidean graph."""
        graph = GraphGenerator.add_edges_erdosrenyi(n, p)
        graph2 = graph.copy(copy_edges = False)
        graph2.q += 1

        for e in graph:
            v, w = e.v, e.w
            costs = [GraphGenerator.euclidean_distance(graph.points[v - 1], graph.points[w - 1])]
            graph2.add_edge(Edge(v, w, costs), False)

        return graph2
