"""This module contains utility functions which serve other modules
"""
import operator
import os.path

import numpy as np
import pandas as pd
import simplejson
from bs4 import BeautifulSoup

first_item = operator.itemgetter(0)


def convet_to_list(file_data):
    if isinstance(file_data, str):
        return eval(file_data)
    return file_data


def post_process(df: pd.DataFrame) -> pd.DataFrame:
    if 'file_data' in df:
        df['file_data'] = df['file_data'].apply(convet_to_list)
    return df


def load_data_set(file_path: str) -> pd.DataFrame:
    """Loads a CSV file using Pandas. Adds a .csv to the file name if the filename extension is not csv.
    When loading the file, check to see if data types for the given file were also saved by looking for a file named
    dtypes_{file_name}.csv. If such file is found, it loads the data set with the dtypes listed in
    dtypes_{file_name}.csv.

    :param file_path: a full path for the file to load
    :return: Pandas FataFrame
    """
    file_path = file_path if '.csv' in file_path else f'{file_path}.csv'
    dtypes = load_dtypes(file_path)
    if not dtypes.empty:
        dtypes_dict = dict(dtypes.to_numpy())
        return post_process(pd.read_csv(file_path, index_col=0, dtype=dtypes_dict))
    else:
        return post_process(pd.read_csv(file_path, index_col=0))


def load_dtypes(file_path: str) -> pd.DataFrame:
    """Load the data types listed in the file loaded from file_path

    :param file_path: the path to the file containing the dtypes
    :return: pd.DataFrame containing the dtypes
    """
    file_name = os.path.basename(file_path)
    dtypes_path = file_path.replace(file_name, f'dtypes_for_{file_name}')
    try:
        return pd.read_csv(dtypes_path)
    except FileNotFoundError:
        return pd.DataFrame()


def save_to_file(data: pd.DataFrame, file_path: str) -> None:
    """Saves a dataframe to the location given by file_path"""
    with open(file_path, 'w') as json_file:
        simplejson.dump(data, json_file)


def save_data_set(data_set: pd.DataFrame, data_folder: str, file_name: str) -> None:
    """Saves a given dataframe as a CSV file

    :param data_set: dataframe to save
    :param data_folder: name of the folder to save the file in
    :param file_name: name of the file, if the name does not contain the extension .csv, it is added to the name
    :return: None
    """
    extension = '' if '.csv' in file_name else '.csv'
    file_name = f'{file_name}{extension}'
    dtype_file_name = f'dtypes_for_{file_name}'

    full_path = os.path.join(data_folder, file_name)
    print(f'saving file into path :{full_path}')
    data_set.to_csv(full_path)

    dtype_full_path = os.path.join(data_folder, dtype_file_name)
    data_set.dtypes.to_frame().to_csv(dtype_full_path)


def case_id_mapping():
    """Return a dict mapping between the number representing case id and an enumeration of the case id"""
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


def po_from_html(body_text: str):
    if not (isinstance(body_text, str) and body_text):
        return None
    soup = BeautifulSoup(body_text)
    po = soup.strong.text if soup.strong else None
    return po


def is_nan(model):
    if model is None:
        return True

    if isinstance(model, str):
        return not model

    return np.isnan(model)
