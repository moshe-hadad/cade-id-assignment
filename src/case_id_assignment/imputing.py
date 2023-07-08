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


PARSINGS = defaultdict(_extract_key_value,
                       {'sale.order_create': _extract_key_value,
                        'sale.order.line_create': _extract_key_value,
                        'account.invoice_create': _extract_key_value,
                        'account.invoice.line_create': _extract_key_value,
                        'account.payment_create': _extract_key_value})


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
            print(1)

        return X
