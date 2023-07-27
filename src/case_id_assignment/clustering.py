import itertools
import markov_clustering as mc
import networkx as nx


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


def _value(item):
    _, value = item
    return value


def _values(cluster):
    return [_value(item) for item in cluster]


def cluster(data_set):
    nodes, edges = _create_graph_from_matrix(data_set)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    matrix = nx.to_scipy_sparse_array(graph)
    # no_date_inflation = 1.40
    no_date_inflation = 1.4
    result = mc.run_mcl(matrix, inflation=no_date_inflation)  # run MCL with default parameters
    clusters_ids = mc.get_clusters(result)  # get clusters
    clusters = [_ids_to_nodes(cluster, nodes) for cluster in clusters_ids]

    clusters_refined = [_values(cluster) for cluster in clusters]

    return clusters_refined
