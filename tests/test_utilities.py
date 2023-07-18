import numpy as np

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
