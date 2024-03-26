from Graph.generators.GraphGenerator import GraphGenerator
from Graph.algorithms.BFS import BFS
import random


def test_on_random_graphs():
    n = 30
    for _ in range(10):
        graph = GraphGenerator.add_edges_erdosrenyi(n, p = 0.6)  # connected
        bfs = BFS(graph)
        assert len(bfs.get_visited_nodes()) == n
        assert 0 <= bfs.get_maximum_level() < n


def test_on_complete_graphs():
    n = 10
    for _ in range(10):
        graph = GraphGenerator.add_edges_complete(n)
        start = random.randint(1, n)
        bfs = BFS(graph, start = start)
        assert len(bfs.get_visited_nodes()) == n
        assert bfs.get_maximum_level() == 1  # all nodes reachable from the start nodes

        nodes_by_levels = bfs.get_nodes_by_levels()

        # all other nodes are reachable directly and thus we have only two layers
        assert len(nodes_by_levels) == 2

        # only the start node has distance 0
        assert nodes_by_levels[0][0] == start

        # all other nodes have distance 1
        assert len(nodes_by_levels[1]) == (n - 1)


def test_on_line_graph():
    n = 10
    for _ in range(10):
        graph = GraphGenerator.add_edges_line(n)
        bfs = BFS(graph, start = 1, build_tree = True)
        assert len(bfs.get_visited_nodes()) == n
        assert bfs.get_maximum_level() == (n - 1)
        assert bfs.get_tree().is_spanning_tree()
        assert bfs.get_last_visited_node() == n
        assert len(bfs.get_nodes_by_levels()) == n


def test_max_nodes_on_line_graph():
    n = 10
    graph = GraphGenerator.add_edges_line(n)
    for i in range(n):
        bfs = BFS(graph, start = 1, max_visited = i + 1)
        assert len(bfs.get_queued_nodes()) == (i + 1)
