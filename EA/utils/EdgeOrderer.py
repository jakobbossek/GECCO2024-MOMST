"""Class with methods ordering of edges according to different criteria."""
from abc import abstractmethod

from EA.utils.EdgeSampler import do_fast_nondominated_sorting
from Graph.Edge import Edge
import random


class EdgeOrderer:
    """Abstract class defining a single method order."""

    @abstractmethod
    def order(self, edges: list[Edge], lambdas: list[float]) -> list[Edge]:
        """Brings edges in some order."""
        pass


class EdgeOrdererExact(EdgeOrderer):
    """Order exactly with exact comparison-based sorting algorithm."""

    def order(self, edges: list[Edge], lambdas: list[float]) -> list[Edge]:
        """
        Sort edges in exact order of weighted/scalarised edges weights.

        This is done with an exact comparison-based sorting algorithm
        in time Theta(n log n).

        Args:
            edges (list[Edge]): a list of Edge objects.
            lambdas (list[float]): a list of weights that add up to one.

        Returns:
            list(Edge): the 'edges' list in sorted order of weighted edge weights.
        """
        return sorted(edges, key=lambda e: e.get_cost(lambdas))


class EdgeOrdererRandom(EdgeOrderer):
    """Random order of edges."""

    def order(self, edges: list[Edge], lambdas: list[float]) -> list[Edge]:
        """
        Shuffle edges randomly.

        Note: the argument 'lambdas' is ignored by this method.

        Args:
            edges (list[Edge]): a list of Edge objects.
            lambdas (list[float]): a list of weights that add up to one.
                Ignored by this method.

        Returns:
            list(Edge): the 'edges' list in sorted order of weighted edge weights.
        """
        random.shuffle(edges)
        return edges


class EdgeDominanceOrder(EdgeOrderer):
    """Return a list of non-dominated edges (based on cost vectors)"""

    def order(self, edges: list[Edge], lambdas: list[float] | None) -> list[Edge]:
        """
        Return those edges, which are non-dominated (based on cost vectors).

        Note 1: the argument 'lambdas' is ignored by this method

        Note 2: the returned list only contains the non-dominated edges! Dominated edges are not contained.

        Args:
            edges (list[Edge]): a list of Edge objects.
            lambdas (list[float]): a list of weights that add up to one.
                Ignored by this method.

        Returns:
            list(Edge): the 'edges' list reduced to the non-dominated edges.

        """
        edge_weights = [e.get_cost() for e in edges]
        ranks, _ = do_fast_nondominated_sorting(edge_weights)

        res_edges = [edges[i] for i in range(len(edges)) if ranks[i] == 1]

        return res_edges
