"""This module holds utilities function for all tests"""
import os

import case_id_assignment.utilities as util


def load_sample_data():
    data_folder = '../data_for_tests'
    data_sample_path = os.path.join(data_folder, 'data_sample.csv')
    data_set_sample = util.load_data_set(file_path=data_sample_path)
    return data_set_sample
