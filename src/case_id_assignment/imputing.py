import itertools
import operator
from collections import defaultdict

import numpy as np
import xml.etree.ElementTree as et

import pandas as pd
import simplejson
from tqdm import tqdm

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
        tqdm.pandas(desc=f'Collect values to imputer: {self.func.__name__}')
        results = X.progress_apply(self.func, axis=1)
        self.to_fill = results[results.notnull()]
        return self

    def transform(self, X, y=None):
        items = list(self.to_fill.items())  # convert from iterator to list so the progress bar will show a bar
        for index, items in tqdm(items, desc=f'Impute values from {self.func.__name__}'):
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
        if 'WH/IN' in po_str:
            return None
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

    if 'WH/IN' in po_str:
        return None
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


class ImputeFromStreamIndexHTTP(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        filtered_ds = X[X['HighestLayerProtocol'] == 'http']
        X = self._sync_on_stream_index(data_set=X, filtered_ds=filtered_ds)
        return X

    def _sync_on_stream_index(self, data_set, filtered_ds):
        excluded = {'file_data', 'stream_index'}
        columns = list(column for column in data_set.columns if column not in excluded)
        for stream_index in filtered_ds['stream_index'].unique():
            filtered = filtered_ds[filtered_ds['stream_index'] == stream_index]
            data_set.update(filtered[columns].ffill().bfill())
        return data_set


class ImputeFromSimilarColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.list_of_columns = None

    def fit(self, X, y=None):
        columns_with_similar_values = util.columns_with_similar_values(X,
                                                                       skip_columns={'file_data', 'real_case_id'})
        self.list_of_columns = columns_with_similar_values
        return self

    def transform(self, X, y=None, **fit_params):
        return self._fill_missing_values(data_set=X)

    def _fill_missing_values(self, data_set):
        for collection_of_columns in self.list_of_columns:
            for first_column, second_column in itertools.permutations(collection_of_columns, 2):
                data_set[first_column] = data_set[first_column].fillna(data_set[second_column])

        return data_set


def _extreact_from_request_method(data_set):
    def extract_request_method_for_impute(row):
        request_method_call, starting_frame_number, frame_number, attribute_name = row[[
            'request_method_call', 'starting_frame_number', 'frame.number', 'attribute_name']]
        value = None

        if request_method_call == 'execute_kw':
            request_method_call_column = data_set[data_set['starting_frame_number'] == str(frame_number)][
                'request_method_call']
            value = None if request_method_call_column.empty else request_method_call_column.iloc[0]
            value = value if value and value.isnumeric() else None


        elif isinstance(request_method_call, str) and request_method_call.isnumeric() and isinstance(
                starting_frame_number, str):
            attribute_column = data_set[data_set['frame.number'] == int(starting_frame_number)]['attribute_name']
            attribute_name = None if attribute_column.empty else attribute_column.iloc[0]
            value = request_method_call

        return {attribute_name: float(value)} if attribute_name and value else None

    return extract_request_method_for_impute


attribute_name = operator.itemgetter(4)


def _extract_attribute_name(value: str):
    return f'{attribute_name(value)}_id'.replace('.', '_') if len(
        value) > 4 else None


class ImputeFromRequestMethodCall(Imputer):
    def __init__(self):
        super().__init__(func=_extreact_from_request_method)

    def fit(self, X, y=None):
        tqdm.pandas(desc=f'Collect values to imputer: ImputeFromRequestMethodCall')
        X['attribute_name'] = X['file_data'].progress_apply(_extract_attribute_name)
        results = X.progress_apply(_extreact_from_request_method(X), axis=1)
        self.to_fill = results[results.notnull()]
        del X['attribute_name']
        return self


class ImputeFromSimilarRows(BaseEstimator, TransformerMixin):
    def __init__(self, columns, excluded_columns=None):
        self.columns = columns
        self.excluded_columns = [] or excluded_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        columns_to_exclude = set(self.excluded_columns)
        columns_for_sync = [column for column in X.columns if column not in columns_to_exclude]
        for column in tqdm(self.columns, desc='Impute from similar rows based on case id attributes'):
            for value in X[column].unique():
                filtered = X[X[column] == value]
                X.update(filtered[columns_for_sync].ffill().bfill())
        return X
