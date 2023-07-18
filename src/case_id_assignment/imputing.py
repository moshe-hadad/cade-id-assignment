import operator
from collections import defaultdict

import numpy as np
import xml.etree.ElementTree as et

import pandas as pd
import simplejson
import case_id_assignment.utilities as util
from sklearn.base import BaseEstimator, TransformerMixin


def _extract_key_value(items):
    it = iter(items[6:])
    return dict(zip(it, it))


def _extract_for_button_confirmed(items):
    id = items[-1]
    return {'res_id': id, 'record_name': f'PO00{id}', 'datas_fname': f'PO_PO00{id}.pdf', 'res_name': f'PO00{id}',
            'origin': f'PO00{id}', 'purchase_order_id': id}


PARSINGS = defaultdict(_extract_key_value,
                       {'sale.order_create': _extract_key_value,
                        'sale.order.line_create': _extract_key_value,
                        'account.invoice_create': _extract_key_value,
                        'account.invoice.line_create': _extract_key_value,
                        'account.payment_create': _extract_key_value,
                        'purchase.order_button_confirm': _extract_for_button_confirmed,
                        'purchase.requisition_create': _extract_key_value,
                        'purchase.requisition.line_create': _extract_key_value,
                        'purchase.order_create': _extract_key_value, })


def parse_null(file_data):
    return np.NAN


ACTION_NAME = operator.itemgetter(4, 5)


def parse_file_data(row):
    try:
        file_data = row['file_data']
        file_data = eval(file_data) if isinstance(file_data, str) else file_data
        if file_data:
            command = util.first_item(file_data)
            if command == 'execute_kw':
                action = '_'.join(ACTION_NAME(file_data))
                parse_method = PARSINGS[action]
                result = parse_method(file_data)
                return result

    except Exception as ex:
        pass
    return np.NAN


def _try_convert_to_int(value):
    if isinstance(value, str) and value.isnumeric():
        return float(value)
    return value


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func
        self.to_fill = None

    def fit(self, X, y=None):
        results = X.apply(self.func, axis=1)
        self.to_fill = results[results.notnull()]
        return self

    def transform(self, X, y=None):
        for index, items in self.to_fill.items():
            columns = list(items.keys())
            values = list(items.values())
            values = [_try_convert_to_int(value) for value in values]
            X.loc[index, columns] = values

        return X


class ImputeFromFileData(Imputer):

    def __init__(self):
        super().__init__(parse_file_data)

    def fit(self, X, y=None):
        results = X.apply(parse_file_data, axis=1)
        self.to_fill = results[results.notnull()]
        return self


def _id_from_query_equals(query):
    query_split = query.split('=')
    id = (query_split[-1].strip())
    return id


def _id_from_query_in(query):
    start_index = query.index('WHERE id IN (') + len('WHERE id IN (')
    end_index = query.rindex(')')
    id = query[start_index: end_index]
    return id


def _extract_id_from_query(query):
    if 'WHERE id=' in query:
        id = _id_from_query_equals(query)
    else:
        id = _id_from_query_in(query)

    return id


def _deal_with_double_id(id):
    if ',' in id:
        ids = id.split(',')
        id = util.first_item(ids)
    return id


def parse_table_column(row):
    query_type = row['query_type']
    if query_type != 'UPDATE':
        return np.NAN

    query = row['query']
    id = _extract_id_from_query(query)
    id = _deal_with_double_id(id)
    tables = row['tables']
    tables = eval(tables)
    table = util.first_item(tables)

    return {f'{table}_id': float(id)}


class ImputeFromTable(Imputer):
    def __init__(self):
        super().__init__(parse_table_column)


def parse_html_columns(row):
    body, body_html = row[['body', 'body_html']]
    po_from_body = util.po_from_html(body)
    po_from_body_html = util.po_from_html(body_html)
    if not po_from_body and not po_from_body_html:
        return None

    po_id = _to_numeric(po_from_body, po_from_body_html)

    return {'purchase_order_id': float(po_id)}


def _str_to_po(po_str):
    if isinstance(po_str, str) and po_str:
        return float(_clean_po_str(po_str))
    else:
        return None if np.isnan(po_str) else po_str


def _clean_po_str(po_str):
    return po_str.replace('Ref ', '').replace('Re: ', '').replace('PO', '').replace('YourCompany Order (', '').replace(
        ')', '').replace('RFQ_', '').replace('PO_)', '').replace('.pdf', '').replace('_', '')


def _to_numeric(po_from_body, po_from_body_html):
    po_str = None
    if po_from_body and po_from_body_html:
        if po_from_body != po_from_body_html:
            print(f'po_from_body: {po_from_body} does not equal to po_from_body_html:{po_from_body_html}')
        po_str = po_from_body
    else:
        po_str = po_from_body or po_from_body_html

    po_id = _str_to_po(po_str)
    return po_id


class ImputeFromHtml(Imputer):

    def __init__(self):
        super().__init__(parse_html_columns)


def extract_po(columns):
    def extract(row):
        for column in columns:
            po_str = row[column]
            po_id = _str_to_po(po_str)
            if po_id:
                break

        return {'purchase_order_id': po_id}

    return extract


class ImputePO(Imputer):
    def __init__(self, columns):
        super().__init__(extract_po(columns))


def extract_po_from_res_id(row):
    res_id, res_model, model = row[['res_id', 'res_model', 'model']]
    model = res_model or model
    if util.is_nan(model):
        return None

    key = f"{model.replace('.', '_')}_id"
    return {key: res_id}


class ImputeFromRes(Imputer):
    def __init__(self):
        super().__init__(extract_po_from_res_id)
