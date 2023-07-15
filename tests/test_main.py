import pytest
from pandas._testing import assert_frame_equal

import case_id_assignment.main as main
import case_id_assignment.utilities as util
from . import testutils as tu


def test_pre_processing_data():
    isolated_sample_data = tu.load_sample_data(file_name='isolated_sample_data.csv')
    interleaved_sample_data = tu.load_sample_data()
    isolated_actual_processed, interleaved_actual_processed = main.pre_processing_data(
        isolated_data_set=isolated_sample_data,
        interleaved_data_set=interleaved_sample_data)

    # util.save_data_set(data_set=isolated_actual_processed, data_folder='../data_for_tests',
    #                    file_name='isolated_expected_processed.csv')
    # util.save_data_set(data_set=interleaved_actual_processed, data_folder='../data_for_tests',
    #                    file_name='interleaved_expected_processed.csv')
    _assert_against(isolated_actual_processed, interleaved_actual_processed,
                    'isolated_expected_processed.csv', 'interleaved_expected_processed.csv')


def _assert_against(isolated_actual, interleaved_actual, isolated_expected_name, interleaved_expected_name):
    isolated_expected = tu.load_sample_data(file_name=isolated_expected_name)
    interleaved_expected = tu.load_sample_data(file_name=interleaved_expected_name)
    assert_frame_equal(isolated_actual, isolated_expected)
    assert_frame_equal(interleaved_actual, interleaved_expected)


def test_engineer_features():
    isolated_processed = tu.load_sample_data(file_name='isolated_expected_processed.csv')
    interleaved_processed = tu.load_sample_data(file_name='interleaved_expected_processed.csv')

    isolated_actual_engineered, interleaved_actual_engineered = main.engineer_features(
        isolated_df_processed=isolated_processed, interleaved_df_processed=interleaved_processed)

    # util.save_data_set(data_set=isolated_actual_engineered, data_folder='../data_for_tests',
    #                    file_name='isolated_expected_engineered.csv')
    # util.save_data_set(data_set=interleaved_actual_engineered, data_folder='../data_for_tests',
    #                    file_name='interleaved_expected_engineered.csv')

    _assert_against(isolated_actual_engineered, interleaved_actual_engineered,
                    'isolated_expected_engineered.csv', 'interleaved_expected_engineered.csv')
