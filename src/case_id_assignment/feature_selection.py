"""Responsible for performing all the necessary actions to perform a correlation check between a given columns and all
other columns.
"""
import math
import statistics
from pprint import pprint

import numpy as np
import pandas as pd
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


def simple_correlation_selector(data_set: pd.DataFrame, target_column: str, threshold: int) -> list[str]:
    """Calculate correlation between all columns and the target_column
    Calculate the mean of the values' count
    Return a list of columns with correlation to target higher than threshold and their values' count in less than the
    calculated mean

    :param data_set: The data frame to run the selection on
    :param target_column: The target column to calculate correlation
    :param threshold: The correlation threshold
    :return: a list of selected columns which are highly correlated to the target columns
    """
    correlation = {}
    counts = []
    for column in data_set.columns:
        if target_column == column or column == 'file_data':
            continue
        new_df = data_set[[target_column, column]]
        if data_set[column].dtypes == 'object':
            new_df = new_df.dropna()
            new_df[column] = new_df[column].astype('category').cat.codes
        corr_df = new_df.corr()
        number_of_values = len(new_df[column].unique())
        counts.append(number_of_values)
        corr_number = corr_df[target_column][column]
        correlation[column] = (number_of_values, corr_number)
    # pprint(correlation)
    mean = math.floor(statistics.mean(counts))
    print(mean)
    filtered = _filter_features(correlation, threshold, count_threshold=mean)
    return filtered
