"""This module holds utilities function for all tests"""
import os
import pandas as pd
import case_id_assignment.utilities as util


def load_sample_data(file_name: str = 'interleaved_sample_data.csv') -> pd.DataFrame:
    """Load sample data sets for tests. Search for the given file name in the data_for tests folder
    If no file_name is given, it defaults to the interleaved sample data

    :param file_name: the data set file name to load
    :return: the loaded data set if found in dat_for_tests folder
    """
    data_folder = '../data_for_tests'
    data_sample_path = os.path.join(data_folder, file_name)

    data_set_sample = util.load_data_set(file_path=data_sample_path)
    return data_set_sample
