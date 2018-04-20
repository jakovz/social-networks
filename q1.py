import collections
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from random import random, choice


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
    g = nx.Graph()
    for i in range(1, nodes_num):
        g.add_node(i)
    # connecting each node to k/2 neighbors on the left and right (by IDs)
    nodes_list = [node for node in g.nodes()]
    for j in range(1, (k / 2) + 1):
        to_nodes = nodes_list[j:] + nodes_list[0:j]
        # adding all initial edges
        g.add_edges_from(zip(nodes_list, to_nodes))
    # for each edge (i,j), deleting it and deciding if it should be recreated with another node with probability of p
    for j in range(1, (k / 2) + 1):
        # iterating through all the edges
        to_nodes = nodes_list[j:] + nodes_list[0:j]
        for node_from, node_to in zip(nodes_list, to_nodes):
            if random() <= p:
                new_node = choice(nodes_list)
                while g.has_edge(node_from, new_node) or new_node == node_from:
                    new_node = choice(nodes_list)
                g.remove_edge(node_from, node_to)
                g.add_edge(node_from, new_node)
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
    :param node: g.node
    :return: node clustering coefficient
    """
    node_neighbors = g.neighbors(node)  # returns iterator
    node_neighbors = [neighbor for neighbor in node_neighbors]
    num_edges_between_neighbors = 0
    node_degree = g.degree(node)
    for edge in g.edges():
        if edge[0] in node_neighbors and edge[1] in node_neighbors:
            num_edges_between_neighbors += 1
    if node_degree < 2:
        clustering_coefficient = 0
    else:
        clustering_coefficient = (2 * float(num_edges_between_neighbors)) / (node_degree * (node_degree - 1))
    return clustering_coefficient


def get_diameter(graph):
    """
    returns graph diameter
    :param graph: nx.Graph()
    :return: graph diameter
    """
    max_diameter = 0
    if not nx.is_connected(graph):
        subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
        for component in subgraphs:
            max_diameter = max(nx.algorithms.distance_measures.diameter(component), max_diameter)
    else:
        max_diameter = nx.algorithms.distance_measures.diameter(graph)
    return max_diameter


def show_degree_distribution(graph):
    """
    print an histogram of degree distribution in a given graph
    :param graph: nx.Graph()
    """
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)


if __name__ == '__main__':
    """
    this code was written to produce the question answers
    """
    erdos_graph = erdos_renyi_model(1000, 0.2)
    small_world_graph = small_world(1000, 8, 0.1)
    print ("Erdos-Renyi graph diameter is:")
    print (get_diameter(erdos_graph))
    print ("Erdos-Renyi graph clustering coefficient is:")
    print (get_graph_clustering_coefficient(erdos_graph))
    print ("Erdos-Renyi graph degree distribution is:")
    print (show_degree_distribution(erdos_graph))
    print ("small world graph diameter is:")
    print (get_diameter(small_world_graph))
    print ("small world graph clustering coefficient is:")
    print (get_graph_clustering_coefficient(small_world_graph))
    show_degree_distribution(small_world_graph)
    nx.draw(small_world_graph)
    plt.draw()
    plt.show()
