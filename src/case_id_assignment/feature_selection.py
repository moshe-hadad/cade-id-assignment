"""Responsible for performing all the necessary actions to perform a correlation check between a given columns and all
other columns.
"""
from pprint import pprint

import numpy as np


def _filter_features(correlation, threshold):
    features = []
    for feature_name, values in correlation.items():
        count, corr = values
        if np.isnan(corr):
            continue
        if corr > threshold:
            features.append(feature_name)

    return features


def simple_correlation_selector(data_set, target_column, threshold):
    correlation = {}
    for column in data_set.columns:
        if target_column == column:
            continue
        new_df = data_set[[target_column, column]]
        if data_set[column].dtypes == 'object':
            new_df = new_df.dropna()
            new_df[column] = new_df[column].astype('category').cat.codes
        corr_df = new_df.corr()
        number_of_values = len(new_df[column].unique())
        corr_number = corr_df[target_column][column]
        correlation[column] = (number_of_values, corr_number)
    filtered = _filter_features(correlation, threshold)
    return filtered
