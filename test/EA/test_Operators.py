from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from Graph.utils.ConstraintChecks import get_degree_constraint_check
from EA.Operators import InsertionFirstGlobalMutator
from EA.Operators import MergeCrossover
from EA.Operators import EdgeExchangeMutator
from EA.Operators import SGSECJMutator
from EA.Operators import SGSImprovedMutator
from EA.Operators import SGSSimplifiedMutator
from EA.Operators import USGMutator
from EA.Operators import USGMutatorWithPresorting
from EA.Operators import RandomWalkMutator
from EA.utils.EdgeSampler import EdgeSamplerUniform, EdgeSamplerDominationCount
import random


def test_SGSECJMutator_on_random_graphs():
    n = 30
    for method in ['bfs', 'dfs']:
        for _ in range(15):
            graph = GraphGenerator.generate_kc_random(n, p = 0.6)
            tree = RandomSpanningTreeGenerator().sample(graph)
            operator = SGSECJMutator(graph, sigma = int(n / 3), traversal_method = method)
            child = operator.create(tree)
            assert child.is_spanning_tree()
            assert 3 <= operator.get_stats()['no_selected_nodes'] <= (n - 1)


def test_SGSImprovedMutator_on_random_graphs():
    n = 30
    for method in ['bfs', 'dfs']:
        for _ in range(15):
            graph = GraphGenerator.generate_kc_random(n, p = 0.6)
            tree = RandomSpanningTreeGenerator().sample(graph)
            operator = SGSImprovedMutator(graph, sigma = int(n / 3), traversal_method = method)
            child = operator.create(tree)
            assert child.is_spanning_tree()
            assert 3 <= operator.get_stats()['no_selected_nodes'] <= (n - 1)

def test_SGSSimplified_on_random_graphs():
    n = 30
    for method in ['bfs', 'dfs']:
        for _ in range(15):
            graph = GraphGenerator.generate_kc_random(n, p = 0.6)
            tree = RandomSpanningTreeGenerator().sample(graph)
            operator = SGSSimplifiedMutator(graph, sigma = int(n / 3), traversal_method = method)
            child = operator.create(tree)
            assert child.is_spanning_tree()
            assert 3 <= operator.get_stats()['no_selected_nodes'] <= (n - 1)

def test_USGMutator_on_random_graphs():
    n = 30
    for _ in range(15):
        for filter_edges in [True, False]:
            graph = GraphGenerator.generate_kc_random(n, p = 0.6)
            tree = RandomSpanningTreeGenerator().sample(graph)
            sigma = random.randint(5, 10)
            operator = USGMutator(graph, sigma = sigma, filter_edges = filter_edges)
            child = operator.create(tree)
            assert child.is_spanning_tree()
            assert 1 <= operator.get_stats()['no_dropped_edges'] <= sigma

def test_USGMutatorWithPresorting_on_random_graphs():
    n = 30
    for _ in range(15):
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree = RandomSpanningTreeGenerator().sample(graph)
        sigma = random.randint(5, 10)
        operator = USGMutatorWithPresorting(graph, sigma = sigma, L = 5)
        child = operator.create(tree)
        assert child.is_spanning_tree()
        assert 1 <= operator.get_stats()['no_dropped_edges'] <= sigma

def test_InsertionFirstGlobalMutator_on_random_graphs():
    n = 30
    for _ in range(50):
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree = RandomSpanningTreeGenerator().sample(graph)
        sigma = 4
        operator = InsertionFirstGlobalMutator(graph, sigma = sigma)
        child = operator.create(tree)
        assert child.is_spanning_tree()
        assert 1 <= operator.get_stats()['no_added_edges'] <= sigma


def test_InsertionFirstGlobalMutator_on_random_graphs_with_degree_constraint():
    n = 30
    for _ in range(50):
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree = RandomSpanningTreeGenerator().sample(graph)

        # use the parents maximum degree plus 1 as degree constraint
        max_degree = tree.get_max_degree() + 1
        sigma = 15

        operator = InsertionFirstGlobalMutator(
            graph,
            sigma = sigma,
            would_violate_constraint = get_degree_constraint_check(max_degree)
        )

        child = operator.create(tree)
        assert child.is_spanning_tree()
        assert not child.violates_degree_constraint(max_degree)
        assert 1 <= operator.get_stats()['no_added_edges'] <= sigma


def test_InsertionFirstGlobalMutator_on_random_graphs_with_biased_edge_sampling_strategy():
    n = 30
    graph = GraphGenerator.generate_kc_random(n, p = 0.6)
    tree = RandomSpanningTreeGenerator.sample(graph)
    biased_edge_sampler = EdgeSamplerDominationCount(graph.get_edges())

    for _ in range(25):
        operator = InsertionFirstGlobalMutator(graph, sigma = 15, edge_sampler = biased_edge_sampler)
        child = operator.create(tree)
        assert child.is_spanning_tree()


def test_EdgeExchangeMutator_on_random_graphs():
    n = 30
    for edge_sampling_strategy in [EdgeSamplerUniform, EdgeSamplerDominationCount]:
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        parent = RandomSpanningTreeGenerator().sample(graph)
        edge_sampler = edge_sampling_strategy(graph.get_edges())

        for _ in range(25):
            operator = EdgeExchangeMutator(edge_sampler, random.randint(1, 4))
            child = operator.create(parent)
            assert child.is_spanning_tree()
            assert 0 <= operator.get_stats()['no_cycle_edges'] <= n


def test_MergeCrossover_on_random_graphs():
    n = 30
    for _ in range(50):
        # We are doing this with two random graphs instead of trees here
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree1 = RandomSpanningTreeGenerator().sample(graph)
        tree2 = RandomSpanningTreeGenerator().sample(graph)

        operator = MergeCrossover(graph)
        child = operator.create(tree1, tree2)
        assert child.is_spanning_tree()


def test_MergeCrossover_on_random_graphs_with_degree_constraint():
    n = 30
    for _ in range(50):
        # We are doing this with two random graphs instead of trees here
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree1 = RandomSpanningTreeGenerator.sample(graph)
        tree2 = RandomSpanningTreeGenerator.sample(graph)

        # fix maximum degree such that both trees are feasible
        max_degree = max(tree1.get_max_degree(), tree2.get_max_degree())

        operator = MergeCrossover(graph, would_violate_constraint = get_degree_constraint_check(max_degree))
        child = operator.create(tree1, tree2)
        assert child.is_spanning_tree()
        assert not child.violates_degree_constraint(max_degree)


def test_RandomWalkMutator_on_random_graphs():
    n = 30
    for _ in range(50):
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree = RandomSpanningTreeGenerator().sample(graph)

        sigma = random.randint(5, 10)
        operator = RandomWalkMutator(graph, sigma = sigma)
        child = operator.create(tree)
        assert child.is_spanning_tree()


def test_RandomWalkMutator_on_random_graphs_with_degree_constraint():
    n = 30
    for _ in range(50):
        graph = GraphGenerator.generate_kc_random(n, p = 0.6)
        tree = RandomSpanningTreeGenerator().sample(graph)

        max_degree = tree.get_max_degree()

        sigma = random.randint(5, 10)
        operator = RandomWalkMutator(graph, sigma = sigma,
                                     would_violate_constraint = get_degree_constraint_check(max_degree))
        child = operator.create(tree)
        assert child.is_spanning_tree()
        assert not child.violates_degree_constraint(max_degree)
