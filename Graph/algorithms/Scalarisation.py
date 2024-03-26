"""Weighted sum scalarisation approches for the moMST."""
from Graph.Graph import Graph
from Graph.Edge import Edge
from Graph.algorithms.SingleObjectiveSolverInterface import SingleObjectiveSolverInterface
from Graph.algorithms.MultiObjectiveSolverInterface import MultiObjectiveSolverInterface
from Graph.algorithms.MST import Kruskal
from typing import Callable
import pyrandvec


class MOMSTWeightedSum(MultiObjectiveSolverInterface):
    """
    mc-{Prim,Kruskal} for the calculation of efficient supported spanning trees.

    Spanning trees with constraints: supported.

    Args:
        graph (Graph): the source graph.
        n_lambdas (int|None): the number of weights to be used. If not None weights
            are sampled by pyrandvec.sample() with method='simplex'.
        lambdas (list[list]|None): list of weight vectors (each summing up to one).
        solver (SingleObjectiveSolverInterface): instance of a single-objective MST
            algorithm.
        would_violate_constraint (function(Graph, Edge)): function that expects a
            Graph (tree) and an Edge object e and returns a Boolean. A return value
            True means that adding e to tree would violate a constraint on the tree.
            Default is no constraint, i.e., a function that always returns False.
    """

    def __init__(self,
                 graph: Graph,
                 n_lambdas: int | None = None,
                 lambdas: list[list] | None = None,
                 solver: SingleObjectiveSolverInterface | None = None,
                 would_violate_constraint: Callable[[Graph, Edge], bool] = (lambda tree, e: False)):
        assert graph.is_connected()

        if n_lambdas is None and lambdas is None:
            raise ValueError('[MOMSTWeightedSum] One of n_lambdas and lambdas must be passed.')

        # generate random weights if not explicitly passed
        if lambdas is None:
            lambdas = pyrandvec.sample(n_lambdas, graph.get_q(), method = 'simplex')

        if solver is None:
            solver = Kruskal(graph, would_violate_constraint = would_violate_constraint)

        self.graph = graph
        self.n_lambdas = n_lambdas
        self.lambdas = lambdas
        self.solver = solver
        self.would_violate_constraint = would_violate_constraint

    def solve(self) -> MultiObjectiveSolverInterface:
        """Calculate approximation of Pareto-set."""
        self.trees = [self.solver.solve(weight).get_solution() for weight in self.lambdas]

        # filter unsuccesful constructions (only for the constraint case)
        self.trees = [tree for tree in self.trees if tree is not None]

        return self

    def get_approximation_set(self) -> list[Graph]:
        """Return set of found non-dominated spanning trees."""
        return self.trees

    def get_approximation_front(self) -> list[list[float]]:
        """Return the set of objective vectors of found non-dominated spanning trees."""
        return [tree.get_sum_of_edge_costs() for tree in self.trees]
