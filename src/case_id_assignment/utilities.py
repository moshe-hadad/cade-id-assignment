"""This module contains utility functions which serve other modules
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


def save_data_set(data_set: pd.DataFrame, data_folder: str, name: str) -> None:
    """Saves a given dataframe as a CSV file

    :param data_set: dataframe to save
    :param data_folder: name of the folder to save the file in
    :param name: name of the file, if the name does not contain the extension .csv, it is added to the name
    :return: None
    """
    extension = '' if '.csv' in name else '.csv'
    name = f'{name}{extension}'
    full_path = os.path.join(data_folder, name)
    print(f'saving file into path :{full_path}')
    data_set.to_csv(full_path)


def case_id_mapping():
    return {399: '{0}',
            400: '{1}',
            401: '{2}',
            402: '{3}',
            403: '{4}',
            404: '{5}',
            405: '{6}',
            406: '{7}',
            407: '{8}',
            408: '{9}'}
