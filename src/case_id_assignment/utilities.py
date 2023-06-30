"""This module contains utilities function which serve all other modules
"""
import operator
import os.path

import pandas as pd

first_item = operator.itemgetter(0)


def load_data_set(file_path: str) -> pd.DataFrame:
    """Loads a CSV file using Pandas

    :param file_path: a full path for the file to load
    :return: Pandas FataFrame
    """
    return pd.read_csv(file_path)


def save_data_set(data_set, data_folder, name):
    full_path = os.path.join(data_folder, name, '.csv')
    data_set.to_csv(full_path)
