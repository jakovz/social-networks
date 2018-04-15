import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from random import random


def erdos_renyi_model(nodes_num, p):
    """
    implementation of erdos-renyi model
    :param nodes_num: number of nodes in the graph
    :param p: probability for each edge to appear in the graph
    :return: nx.Graph()
    """
    g = nx.Graph()
    for i in range(1, nodes_num):
        g.add_node(i)
    for edge in combinations(range(1, nodes_num), 2):
        # combinations(range(1,nodes_num), 2) returns all possible pairs from the given range
        if random() <= p:
            g.add_edge(*edge)
    return g


def small_world(nodes_num, k, p):
    """
    implementation of watts-strogatz model
    :param nodes_num: number of nodes in the graph
    :param k: average node degree in the graph
    :param p: probability for recreating the edge
    :return: nx.Graph()
    """
    # should return a random graph here
    g = nx.Graph()
    for i in range(1, nodes_num):
        g.add_node(i)
    # connecting each node to k/2 neighbors on the left and right (by IDs)
    # the source of the algorithm is taken from: https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    for i in range(1, nodes_num):
        for j in range(i, k):
            if abs(i - j) % (nodes_num - 1 - (k / 2)) <= (k / 2):
                g.add_edge(i, i + j)
    # for each edge (i,j), deleting it and deciding if it should be recreated with probability of p
    graph_edges = g.edges.items()
    for edge in graph_edges:
        g.remove_edge(*edge[0])
        if random() <= p:
            g.add_edge(*edge[0])
    return g


def get_graph_clustering_coefficient(graph):
    """
    returns graph clustering coefficient (graph average clustering coefficient)
    :param graph: nx.Graph()
    :return: graph clustering coefficient
    """
    sum_clustering_coefficients = 0
    for node in graph.nodes():
        sum_clustering_coefficients += get_node_clustering_coefficient(graph, node)
    return sum_clustering_coefficients / graph.number_of_nodes()


def get_node_clustering_coefficient(g, node):
    """
    returns the clustering coefficient of the given node
    :param g: nx.Graph()
    :param node: the requested node in the graph
    :return: node clustering coefficient
    """
    node_neighbors = g.neighbors(node)  # returns iterator
    node_neighbors = [neighbor for neighbor in node_neighbors]
    num_edges_between_neighbors = 0
    node_degree = g.degree(node)
    for edge in g.edges():
        if edge[0] in node_neighbors and edge[1] in node_neighbors:
            num_edges_between_neighbors += 1
    if (node_degree<2):
        clustering_coefficient = 0
    else:
        clustering_coefficient = (2 * float(num_edges_between_neighbors)) / (node_degree * (node_degree - 1))
    return clustering_coefficient


def get_diameter(graph):
    max_diameter = 0
    if not nx.is_connected(graph):
        subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
        for component in subgraphs:
            max_diameter = max(nx.algorithms.distance_measures.diameter(component), max_diameter)
    else:
        max_diameter = nx.algorithms.distance_measures.diameter(graph)
    return max_diameter


if __name__ == '__main__':
    erdos_graph = erdos_renyi_model(500, 0.2)
    small_world_graph = small_world(500, 8, 0.2)
    print ("Erdos-Renyi graph diameter is:")
    print (get_diameter(erdos_graph))
    print ("Erdos-Renyi graph clustering coefficient is:")
    print (get_graph_clustering_coefficient(erdos_graph))
    print ("small world graph diameter is:")
    print (get_diameter(small_world_graph))
    print ("small world graph clustering coefficient is:")
    print (get_graph_clustering_coefficient(small_world_graph))
    nx.draw(erdos_graph)
    nx.draw(small_world_graph)
