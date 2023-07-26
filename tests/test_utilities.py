import numpy as np
import pandas as pd

import case_id_assignment.utilities as util


def test_is_nan():
    actual = util.is_nan(None)
    assert actual is True

    actual = util.is_nan(np.NAN)
    assert actual == True

    actual = util.is_nan('')
    assert actual is True

    actual = util.is_nan('test')
    assert actual is False

    actual = util.is_nan(12)
    assert actual == False


def test_missing_data_percentage():
    data = pd.DataFrame(data={
        'column A': [1, 2, np.nan, 4, 5],
        'column B': ['A', 'B', '', 'C', 'B'],
        'column C': [1.2, 3.5, np.nan, 4.5, 3.2],
    })
    actual = util.missing_data_percentage(data)
    assert actual == 0.2


def test_columns_with_similar_values():
    data = pd.DataFrame(data={
        'A': [1, 2, 2, 4, 5],
        'B': ['A', 'B', '', 'C', 'B'],
        'C': [4, 2, np.nan, 1, 5],
        'D': ['B', 'B', np.nan, 'C', 'A'],
        'E': ['C', 'A', np.nan, 'B', 'A'],
        'file_data': [[1, 2, 3], [8, 5, 4], [6, 5, 4], [7, 8, 9], [7, 8, 9], ],
        'F': ['1956', '2156', '1212', '456', '85']
    })
    actual = util.columns_with_similar_values(data_set=data, skip_columns={'file_data'})
    expected = [['A', 'C'], ['B', 'D', 'E']]

    assert actual == expected

#
# def test_fix_benchmark():
#     benchmark = util.load_data_set('../processed_data/benchmark.csv')
#     final = util.load_data_set('../processed_data/final_results.csv')
#     benchmark['real_activity_action'] = final['real_activity_action']
#     util.save_data_set(data_set=benchmark, data_folder='../processed_data', file_name='benchmark.csv')
