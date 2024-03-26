from Graph.generators.GraphGenerator import GraphGenerator
from Graph.algorithms.DFS import DFS
import random


def test_on_random_graphs():
    n = 30
    for _ in range(10):
        graph = GraphGenerator.add_edges_erdosrenyi(n, p = 0.6)  # connected
        dfs = DFS(graph)
        assert len(dfs.get_visited_nodes()) == n


def test_max_nodes_on_line_graph():
    n = 10
    graph = GraphGenerator.add_edges_line(n)
    for i in range(n):
        dfs = DFS(graph, start = 1, max_visited = i + 1, build_tree = True)
        assert len(dfs.get_queued_nodes()) == (i + 1)
        assert dfs.get_tree().is_forest()
