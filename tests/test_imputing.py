import numpy as np

import pandas as pd
from sklearn.pipeline import Pipeline
from pandas.testing import assert_frame_equal
import case_id_assignment.imputing as imp
import case_id_assignment.utilities as util

from . import testutils as tu


def _input_sample_data():
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('inputter', imp.ImputeFromFileData())
    ])
    results = pipeline.fit_transform(sample_data)
    return results


def test_impute_sale_order_create():
    results = _input_sample_data()
    data = {'partner_id': 1, 'partner_invoice_id': 1, 'partner_shipping_id': 1}
    indices = [7, 49, 92]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices, dtype=float)

    assert_frame_equal(actual, expected)


def test_impute_sale_order_line_create():
    results = _input_sample_data()

    data = {'order_id': [94., 103.], 'product_id': [21., 21.], 'name': ['Conference Chair', 'Conference Chair'],
            'product_uom_qty': [1., 3.], 'price_unit': [5., 4.]}
    indices = [20, 62]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices)

    assert_frame_equal(actual, expected)


def test_impute_account_invoice_create():
    results = _input_sample_data()
    data = {'name': ['Cabinet with Doors_203', 'Acoustic Bloc Screens_173'],
            'partner_id': [1243., 1220.], 'currency_id': [2., 2.], 'origin': ['PO00203', 'PO00173']}
    indices = [194, 246]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices)

    assert_frame_equal(actual, expected)


def test_impute_account_invoice_line_create():
    results = _input_sample_data()
    data = {'invoice_id': [38., 39.], 'name': ['Conference Chair', 'Office Chair'],
            'price_unit': [5., 3.], 'quantity': [1., 3.], 'purchase_line_id': [163., 167.], 'account_id': [24., 24.]}
    indices = [188, 199]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices)

    assert_frame_equal(actual, expected)


def test_impute_account_payment_create():
    results = _input_sample_data()
    data = {'payment_type': ['outbound'], 'payment_method_id': [2.], 'amount': [1.],
            'currency_id': [2.], 'journal_id': [2.]}
    indices = [196]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices)

    assert_frame_equal(actual, expected)

