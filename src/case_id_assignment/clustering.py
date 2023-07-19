import itertools
import markov_clustering as mc
import networkx as nx


def _create_node_and_edge(record):
    values = record[record.notnull()]
    if not any(values):
        return [], []
    nodes = [value for _, value in values.items()]
    edges = [(first_value, second_value) for first_value, second_value in itertools.combinations(nodes, 2)]
    return nodes, edges


def _create_graph_from_matrix(data_set):
    data = data_set.to_dict('records')
    nodes = []
    edges = []
    for _, record in data_set.iterrows():
        node, edge = _create_node_and_edge(record)
        nodes.extend(node)
        edges.extend(edge)

    return list(set(nodes)), list(set(edges))


def _ids_to_nodes(cluster, nodes):
    return [nodes[id] for id in cluster]


def cluster(data_set):
    nodes, edges = _create_graph_from_matrix(data_set)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    matrix = nx.to_scipy_sparse_array(graph)
    no_date_inflation = 1.40
    result = mc.run_mcl(matrix, inflation=no_date_inflation)  # run MCL with default parameters
    clusters_ids = mc.get_clusters(result)  # get clusters
    clusters = [_ids_to_nodes(cluster, nodes) for cluster in clusters_ids]
    return clusters
