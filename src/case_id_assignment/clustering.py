import itertools
import markov_clustering as mc
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

import case_id_assignment.utilities as util


def _create_node_and_edge(record):
    values = record[record.notnull()]
    if values.empty:
        return [], []
    nodes = [(key, value) for key, value in values.items()]
    edges = [(first_value, second_value) for first_value, second_value in itertools.combinations(nodes, 2)]
    return nodes, edges


def _create_graph_from_matrix(data_set):
    nodes = []
    edges = []
    for _, record in data_set.iterrows():
        node, edge = _create_node_and_edge(record)
        nodes.extend(node)
        edges.extend(edge)

    return list(set(nodes)), list(set(edges))


def _ids_to_nodes(cluster, nodes):
    return [nodes[id] for id in cluster]


def _value(item: str):
    key, value = item
    return value
    # key, value = item.split('=')
    # return value
    # _, value = item
    # return value


def _values(cluster):
    return [_value(item) for item in cluster]


def cluster(data_set, save_to_file=True) -> list[list[object]]:
    graph, nodes = build_graph(data_set)
    # nx.draw(graph, with_labels=True, font_weight='bold')
    # plt.show()
    matrix = nx.to_scipy_sparse_array(graph)
    # no_date_inflation = 1.40
    no_date_inflation = 1.4
    result = mc.run_mcl(matrix, inflation=no_date_inflation)  # run MCL with default parameters
    clusters_ids = mc.get_clusters(result)  # get clusters

    clusters = [_ids_to_nodes(cluster, nodes) for cluster in clusters_ids]
    data = {'clusters': clusters}
    if save_to_file:
        util.save_data_set(data_set=pd.DataFrame(data=data), data_folder='../../processed_data',
                           file_name='clusters.csv')
    clusters_refined = [_values(cluster) for cluster in clusters]

    return clusters_refined


def build_graph(data_set):
    nodes, edges = _create_graph_from_matrix(data_set)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph, nodes


def girvan_newman(data_set: pd.DataFrame) -> list[list[object]]:
    graph, _ = build_graph(data_set)
    girvan_newman = nx.community.girvan_newman(graph)
    clusters = next(girvan_newman)
    return clusters


def naive_greedy_modularity_communities(data_set: pd.DataFrame) -> list[list[object]]:
    graph, _ = build_graph(data_set)
    clusters = nx.community.naive_greedy_modularity_communities(graph)
    return list(clusters)


def kernighan_lin_bisection(data_set: pd.DataFrame) -> list[list[object]]:
    graph, _ = build_graph(data_set)
    clusters = nx.community.kernighan_lin_bisection(graph)
    return list(clusters)


def louvain_communities(data_set: pd.DataFrame) -> list[list[object]]:
    graph, _ = build_graph(data_set)
    clusters = nx.community.louvain_communities(graph)
    clusters = list(clusters)
    return clusters


def greedy_modularity_communities(data_set: pd.DataFrame, save_to_file=True):
    graph, _ = build_graph(data_set)
    results = nx.community.greedy_modularity_communities(graph)
    if save_to_file:
        data = {'clusters': list(results)}
        util.save_data_set(data_set=pd.DataFrame(data=data), data_folder='../../processed_data',
                           file_name='clusters.csv')
    return results
