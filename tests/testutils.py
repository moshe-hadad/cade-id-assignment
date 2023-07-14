"""This module holds utilities function for all tests"""
import os

import case_id_assignment.utilities as util


def load_sample_data(file_name='sample_data.csv'):
    data_folder = '../data_for_tests'
    data_sample_path = os.path.join(data_folder, file_name)

    data_set_sample = util.load_data_set(file_path=data_sample_path)
    return data_set_sample
