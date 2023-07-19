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
