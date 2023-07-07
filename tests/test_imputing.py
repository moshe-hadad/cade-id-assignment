import numpy as np

import pandas as pd
from sklearn.pipeline import Pipeline
from pandas.testing import assert_frame_equal
import case_id_assignment.imputing as imp
import case_id_assignment.utilities as util

from . import testutils as tu


def test_impute_from_message_attributes():
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('inputter', imp.ImputeFromFileData())
    ])
    results = pipeline.fit_transform(sample_data)
    data = {'partner_id': 1, 'partner_invoice_id': 1, 'partner_shipping_id': 1}
    indices = [7, 49, 92]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices, dtype=float)

    assert_frame_equal(actual, expected)
