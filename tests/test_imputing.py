import numpy as np

import pandas as pd
import pytest
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


def _impute_(sample_data=None, inputer_class=imp.ImputeFromTable()):
    sample_data = sample_data if isinstance(sample_data, pd.DataFrame) else tu.load_sample_data(
        file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('inputter', inputer_class)
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
    results = _impute_()
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

    actual_po = util.po_from_html(np.NAN)
    assert actual_po is None

    actual_po = util.po_from_html(None)
    assert actual_po is None

    actual_po = util.po_from_html(12.0)
    assert actual_po is None


def test_impute_from_html():
    results = _impute_(inputer_class=imp.ImputeFromHtml())
    data = {'purchase_order_id': [156., 156.]}
    indices = [25926., 25927.]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test__to_numeric():
    assert_to_numeric(po_body='PO00982', po_html='PO00982')

    assert_to_numeric(po_body=None, po_html='PO00982')

    assert_to_numeric(po_body='PO00982', po_html=None)


def assert_to_numeric(po_body, po_html):
    actual_po = imp._to_numeric(po_body, po_html)
    assert actual_po == 982.0


def test_impute_po():
    columns = ['subject', 'origin', 'res_name', 'datas_fname']
    results = _impute_(inputer_class=imp.ImputePO(columns))
    data = {'purchase_order_id': [152., 153., 153., 158., 152.]}
    indices = [25928, 25929, 25932, 25933, 25934]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test_impute_from_res_id():
    results = _impute_(inputer_class=imp.ImputeFromRes())
    data = {'purchase_order_id': [156., np.nan], 'sale_order_id': [np.nan, 94.]}
    indices = [25930, 25931]

    actual = results.loc[indices, list(data.keys())]
    expected = expected_results(data, indices)

    assert_frame_equal(actual, expected)


def test__clean_po_str():
    actual = imp._clean_po_str('RFQ_PO00158.pdf')
    assert actual == '00158'

    actual = imp._clean_po_str('PO_PO00152.pdf')
    assert actual == '00152'


def test_impute_from_stream_index():
    data_set = pd.DataFrame({
        'stream_index': [1, 1, 2, 3, 4, 3, 3, 8, 8, 8],
        'HighestLayerProtocol': ['http', 'http', 'pgsql', 'http', 'http', 'http', 'http', 'pgsql', 'pgsql', 'pgsql'],
        'sale_order_line_id': [375, np.nan, np.nan, 376, np.nan, np.nan, np.nan, np.nan, np.nan, 399],
        'order_line_id': [np.nan, 378, np.nan, np.nan, 376, 376, np.nan, np.nan, np.nan, 566],
    })
    actual = _impute_(sample_data=data_set, inputer_class=imp.ImputeFromStreamIndexHTTP())

    expected = pd.DataFrame({
        'stream_index': [1, 1, 2, 3, 4, 3, 3, 8, 8, 8],
        'HighestLayerProtocol': ['http', 'http', 'pgsql', 'http', 'http', 'http', 'http', 'pgsql', 'pgsql', 'pgsql'],
        'sale_order_line_id': [375, 375, np.nan, 376, np.nan, 376, 376, np.nan, np.nan, 399],
        'order_line_id': [378, 378, np.nan, 376, 376, 376, 376, np.nan, np.nan, 566],
    })
    assert_frame_equal(actual, expected)


def test_impute_from_similar_columns():
    """Give a data set with this structure
     attachment | HighestLayerProtocol | sale_order_id | sale_order_line_id | order_line_id | test_mix
     --------------------------------------------------------------------------------------------------
     'PO00978'
     np.nan

    :return: None
    """
    data_set = pd.DataFrame({
        'attachment': ['PO00978', np.nan, 'PO00979', 'PO00980', 'PO00981', 'PO00978', np.nan, 'PO00981'],
        'HighestLayerProtocol': [np.nan, 'PO00979', np.nan, 'PO00980', 'PO00981', np.nan, 'PO00978', np.nan],
        'sale_order_id': [np.nan, np.nan, 375, np.nan, 376, np.nan, 375, 378],
        'sale_order_line_id': [375, np.nan, np.nan, 376, np.nan, 378, np.nan, np.nan],
        'order_line_id': [np.nan, 378, np.nan, np.nan, 376, 378, 375, np.nan],
        'test_mix': [np.nan, '378', np.nan, np.nan, '376', '378', 375, np.nan],
    })

    actual = _impute_(sample_data=data_set, inputer_class=imp.ImputeFromSimilarColumns())

    expected = pd.DataFrame({
        'attachment': ['PO00978', 'PO00979', 'PO00979', 'PO00980', 'PO00981', 'PO00978', 'PO00978', 'PO00981'],
        'HighestLayerProtocol': ['PO00978', 'PO00979', 'PO00979', 'PO00980', 'PO00981', 'PO00978', 'PO00978',
                                 'PO00981'],
        'sale_order_id': [375., 378., 375., 376., 376., 378., 375., 378.],
        'sale_order_line_id': [375., 378., 375., 376., 376., 378., 375., 378.],
        'order_line_id': [375., 378., 375., 376., 376., 378., 375., 378.],
        'test_mix': [np.nan, '378', np.nan, np.nan, '376', '378', 375, np.nan],
    })

    assert_frame_equal(actual, expected)


def test_impute_method_call():
    data_set = pd.DataFrame(_data_for_impute_method_call())

    actual = _impute_(sample_data=data_set, inputer_class=imp.ImputeFromRequestMethodCall())

    sale_order_id = [np.nan, np.nan, np.nan, np.nan, np.nan, 375, np.nan, 375, np.nan]
    sale_order_line_id = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 399, np.nan, 399]
    expected = pd.DataFrame(
        _data_for_impute_method_call(sale_order_id=sale_order_id, sale_order_line_id=sale_order_line_id))

    assert_frame_equal(actual, expected)


def _data_for_impute_method_call(sale_order_id=None, sale_order_line_id=None):
    request_method_call = ['version', 'server_version', np.nan, np.nan, np.nan, 'execute_kw', 'execute_kw', '375',
                           '399']
    frame_number = [99, 92, 100, 200, 220, 240, 318, 350, 400]
    starting_frame_number = [np.nan, np.nan, '99', '92', np.nan, np.nan, np.nan, '240', '318']
    sale_order_id = sale_order_id or [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    sale_order_line_id = sale_order_line_id or [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    sale_order = ['execute_kw', 'odoo01', '2', 'PWD1234', 'sale.order', 'create', 'partner_id', '1',
                  'partner_invoice_id', '1', 'partner_shipping_id', '1']
    sale_order_line = ['execute_kw', 'odoo01', '2', 'PWD1234', 'sale.order.line', 'create', 'order_id', '375',
                       'product_id', '6', 'name', 'Office Lamp', 'product_uom_qty', '3', 'price_unit', '2']
    file_data = [[], [], [], [], [], sale_order, sale_order_line, [375], [399]]

    return {
        'request_method_call': request_method_call,
        'frame.number': frame_number,
        'starting_frame_number': starting_frame_number,
        'sale_order_id': sale_order_id,
        'sale_order_line_id': sale_order_line_id,
        'file_data': file_data
    }
