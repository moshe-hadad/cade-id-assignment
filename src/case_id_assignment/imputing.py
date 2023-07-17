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


def parse_file_data(file_data):
    try:
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


class ImputeFromFileData(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.to_fill = None

    def fit(self, X, y=None):
        results = X['file_data'].apply(parse_file_data)
        self.to_fill = results[results.notnull()]
        return self

    def transform(self, X, y=None):
        for index, items in self.to_fill.items():
            columns = list(items.keys())
            values = list(items.values())
            values = [_try_convert_to_int(value) for value in values]
            X.loc[index, columns] = values

        return X


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


class ImputeFromTable(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.to_fill = None

    def fit(self, X, y=None):
        results = X.apply(parse_table_column, axis=1)
        self.to_fill = results[results.notnull()]
        return self

    def transform(self, X, y=None):
        for index, items in self.to_fill.items():
            columns = list(items.keys())
            values = list(items.values())
            values = [_try_convert_to_int(value) for value in values]
            X.loc[index, columns] = values

        return X


def parse_html_columns(row):
    body = row['body']
    body_html = row['body_html']
    po_from_body = util.po_from_html(body)
    po_from_body_html = util.po_from_html(body_html)

    query = row['query']
    id = _extract_id_from_query(query)
    id = _deal_with_double_id(id)
    tables = row['tables']
    tables = eval(tables)
    table = util.first_item(tables)

    return {f'{table}_id': float(id)}


class ImputeFromHtml(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.to_fill = None

    def fit(self, X, y=None):
        results = X.apply(parse_html_columns, axis=1)
        self.to_fill = results[results.notnull()]
        return self

    def transform(self, X, y=None):
        for index, items in self.to_fill.items():
            columns = list(items.keys())
            values = list(items.values())
            values = [_try_convert_to_int(value) for value in values]
            X.loc[index, columns] = values

        return X
