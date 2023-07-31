"""Responsible for performing all the necessary actions to perform a correlation check between a given columns and all
other columns.
"""
import itertools
import math
import statistics
from pprint import pprint

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import case_id_assignment.utilities as util
from sklearn.base import BaseEstimator, TransformerMixin


def _filter_features(correlation, threshold, count_threshold):
    features = []
    for feature_name, values in correlation.items():
        count, corr = values
        if np.isnan(corr):
            continue
        if corr > threshold and count < count_threshold:
            features.append(feature_name)

    return features


def _to_data_set(correlation):
    if isinstance(correlation, dict):
        names = correlation.keys()
        values = correlation.values()
        values_count, correlation = zip(*values)
        data = {'names:': names,
                'correlation': correlation,
                'values_count': values_count}
        correlation_ds = pd.DataFrame(data)
        correlation_ds = correlation_ds.sort_values("correlation", ascending=False)
    else:
        data = {'names': list(correlation)}
        correlation_ds = pd.DataFrame(data)
    return correlation_ds


def simple_correlation_selector(data_set: pd.DataFrame, target_column: str, threshold: int,
                                count_threshold: int = None, save_to_file: bool = True) -> list[str]:
    """Calculate correlation between all columns and the target_column
    Calculate the mean of the values' count
    Return a list of columns with correlation to target higher than threshold and their values' count in less than the
    calculated mean

    :param count_threshold: The count threshold, attrs with unique values above count threshold will not be retrieved
    :param data_set: The data frame to run the selection on
    :param target_column: The target column to calculate correlation
    :param threshold: The correlation threshold
    :return: a list of selected columns which are highly correlated to the target columns
    """
    correlation = {}
    counts = []
    columns = list(data_set.columns)
    if 'file_data' in data_set.columns:
        columns.remove('file_data')
    columns.remove(target_column)
    for column in columns:
        new_df = data_set[[target_column, column]]
        new_df = new_df.dropna(subset=[column])
        if data_set[column].dtypes == 'object':
            new_df[column] = new_df[column].astype('category').cat.codes
        corr_df = new_df.corr()
        number_of_values = len(data_set[column].unique())
        counts.append(number_of_values)
        corr_number = corr_df[target_column][column]
        correlation[column] = (number_of_values, corr_number)
    if save_to_file:
        save_correlation(correlation)
    # pprint(correlation)
    mean = math.floor(statistics.mean(counts))
    count_threshold = count_threshold or mean
    filtered = _filter_features(correlation, threshold, count_threshold=count_threshold)
    return filtered


def save_correlation(correlation):
    correlation_ds = _to_data_set(correlation)
    util.save_data_set(data_set=correlation_ds, data_folder='../../processed_data', file_name='correlation.csv')


def _remove_known_columns(columns):
    columns_to_remove = {'active_id', 'file_data', 'synthetic_sniff_time', 'synthetic_sniff_time_str', 'sniff_time',
                         'FileName', 'body_html', 'message_id'}
    for column in list(columns):
        if 'Unnamed' in column or 'date' in column or column in columns_to_remove:
            columns.remove(column)
    return columns


def unique_cover(data_set: pd.DataFrame, target_column, save_to_file: bool = True):
    cover_columns = set()
    counts = {}
    columns = list(data_set.columns)
    columns = _remove_known_columns(columns)
    columns.remove(target_column)
    for column in tqdm(columns, desc='Select features for case id attributes'):
        new_df = data_set[[target_column, column]]
        new_df = new_df.dropna(subset=[column])
        if new_df.empty:
            continue
        number_of_values = len(data_set[column].unique())
        counts[column] = number_of_values
        cover = []
        for value in new_df[column].unique():
            filtered = new_df[new_df[column] == value][target_column]
            is_unique = len(filtered.unique()) == 1
            cover.append(is_unique)

        # cover = [new_df[new_df[column] == value][target_column].duplicated(keep=False) for value in
        #          new_df[column].unique()]
        all_unique = all(cover)
        if all_unique:
            cover_columns.add(column)
    if save_to_file:
        save_correlation(cover_columns)
    # mean = math.floor(statistics.mean(counts.values()))
    # count_threshold = count_threshold or mean
    # columns_to_filter = [column ]
    return list(cover_columns)

# def select_by_degree(data_set: pd.DataFrame):
#     graph = nx.Graph()
#     # graph.add_nodes_from(nodes)
#     # graph.add_edges_from(edges)
#     for index, row in data_set.iterrows():
#         values = row.dropna()
#         if values.empty:
#             continue
#         for combination in itertools.combinations(values.keys(), 2):
#             first, second = combination
#             graph.add_edge(first, second)
#
#     centrality = nx.degree_centrality(graph)
#     return None
