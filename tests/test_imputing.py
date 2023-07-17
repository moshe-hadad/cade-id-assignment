import numpy as np

import pandas as pd
from sklearn.pipeline import Pipeline
from pandas.testing import assert_frame_equal
import case_id_assignment.imputing as imp
import case_id_assignment.utilities as util

from . import testutils as tu
from .testutils import expected_results


def _input_sample_data():
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('inputter', imp.ImputeFromFileData())
    ])
    results = pipeline.fit_transform(sample_data)
    return results


def _impute_from_table_data():
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('inputter', imp.ImputeFromTable())
    ])
    results = pipeline.fit_transform(sample_data)
    return results


def test_impute_sale_order_create():
    results = _input_sample_data()
    data = {'partner_id': 1., 'partner_invoice_id': 1., 'partner_shipping_id': 1.}
    indices = [7., 49., 92.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_sale_order_line_create():
    results = _input_sample_data()

    data = {'order_id': [94., 103.], 'product_id': [21., 21.], 'name': ['Conference Chair', 'Conference Chair'],
            'product_uom_qty': [1., 3.], 'price_unit': [5., 4.]}
    indices = [20., 62.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_account_invoice_create():
    results = _input_sample_data()
    data = {'name': ['Cabinet with Doors_203', 'Acoustic Bloc Screens_173'],
            'partner_id': [1243., 1220.], 'currency_id': [2., 2.], 'origin': ['PO00203', 'PO00173']}
    indices = [22163., 25925.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_purchase_requisition_create():
    results = _input_sample_data()
    data = {'name': ['Cabinet with Doors_110', 'Acoustic Bloc Screens_102'], 'type_id': [2., 2.]}
    indices = [3031., 4230.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_purchase_requisition_line_create():
    results = _input_sample_data()
    data = {'requisition_id': [100., 101.], 'product_id': [6., 17.], 'product_qty': [1., 3.],
            'price_unit': [3., 4.]}
    indices = [3637., 3658.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_account_invoice_line_create():
    results = _input_sample_data()
    data = {'invoice_id': [38., 39.], 'name': ['Conference Chair', 'Office Chair'],
            'price_unit': [5., 3.], 'quantity': [1., 3.], 'purchase_line_id': [163., 167.], 'account_id': [24., 24.]}
    indices = [21819., 22447.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_account_payment_create():
    results = _input_sample_data()
    data = {'payment_type': ['outbound'], 'payment_method_id': [2.], 'amount': [1.],
            'currency_id': [2.], 'journal_id': [2.]}
    indices = [22326.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_purchase_order_button_confirm():
    results = _input_sample_data()
    data = {'res_id': [152., 156.], 'record_name': ['PO00152', 'PO00156'],
            'datas_fname': ['PO_PO00152.pdf', 'PO_PO00156.pdf'],
            'res_name': ['PO00152', 'PO00156'], 'origin': ['PO00152', 'PO00156'], 'purchase_order_id': [152., 156.]}
    indices = [16075., 16725.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_purchase_order_create():
    results = _input_sample_data()
    data = {'partner_id': [1219., 1219.], 'requisition_id': [101., 103.]}
    indices = [10163., 10416.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_from_table():
    results = _impute_from_table_data()
    data = {'sale_order_id': [94., 94., 94., 94.]}
    indices = [11., 12., 13., 14.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_extract_po_from_html():
    body_text = """<div style=margin:0px; padding:0px>     <p style=margin:0px; padding:0px; font-size:13px>         
    Here is in attachment a purchase order <strong>PO00982</strong><br><br>         
    If you have any questions please do not hesitate to contact us.         <br><br>         
    Best regards </p> </div>"""
    actual_po = util.po_from_html(body_text)
    expected_po = "PO00982"
    assert actual_po == expected_po


def test_impute_from_html():
    results = _impute_from_table_data()
    data = {'purchase_order_id': [156., 156.]}
    indices = [25926., 25927.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)

# def test_create_another_file():
#     df = util.load_data_set('../processed_data/interleaved_df_imputed.csv' '',)
#     new_df = df[:200]
#     util.save_data_set(data_set=new_df, data_folder='../data_for_tests', file_name='interleaved_stream_index_imputing.csv')


# def test_fix_index():
#     sfi = util.load_data_set('./data_for_tests/sample_for_imputing.csv')
#     util.save_data_set(sfi, data_folder='./data_for_tests', file_name='sample_for_imputing.csv')
