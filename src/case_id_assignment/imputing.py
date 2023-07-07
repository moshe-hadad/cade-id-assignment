import operator

import numpy as np
import xml.etree.ElementTree as et

import pandas as pd
import simplejson
import case_id_assignment.utilities as util
from sklearn.base import BaseEstimator, TransformerMixin


def parse_file_data_from_xml(param):
    root = et.fromstring(param)
    filtered = [item for item in list(root.itertext()) if item.strip()]
    return filtered


def impute_from_message_attributes(data_set):
    def extract_values_message_attributes(row):
        session_class = row['session_class']
        message_attributes_is_str = False
        if session_class != 'http':
            return '', None, []
        if isinstance(row['MessageAttributes'], str):
            column_data = row['MessageAttributes'].replace("'", '"').replace('"1.0"', '1.0')
            message_attributes_is_str = True
        # print(column_data)
        xml = None
        starting_frame_number = ''
        try:
            message_attributes = simplejson.loads(column_data) if message_attributes_is_str else row[
                'MessageAttributes']
            starting_frame_number = message_attributes.get('http.request_in', '')
        except simplejson.errors.JSONDecodeError:
            index = column_data.index('"http.file_data":') + len('"http.file_data":') + 2
            xml = column_data[index:-2].encode().decode('unicode_escape')

        try:
            xml = xml if xml else message_attributes['http.file_data']
        except KeyError:
            return 'KeyError'
        xml = xml if '"1.0"' in xml else xml.replace('1.0', '"1.0"')
        xml = xml.replace('\\xa', '')

        return xml

    return extract_values_message_attributes


def _clean_xml_str(file_data):
    # xml = file_data.replace('1.0', '"1.0"').replace('\\xa', '')
    xml = file_data.replace('\\xa', '')
    return xml


def parse_message_attributes(row):
    try:
        message_attributes = eval(row)
        http_type = 'http.content_type'
        if http_type in message_attributes:
            file_data = message_attributes['http.file_data']
            xml = _clean_xml_str(file_data)
            method = parse_file_data_from_xml(xml)
        else:
            return np.NAN

    except Exception as ex:
        pass
    return 'test'


class ImputeFromMessageAttributes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        results = X['MessageAttributes'].apply(parse_message_attributes)
        self.imputer_map = 'non'
        return self

    def transform(self, X, y=None):
        return X


def _extract_key_value(items):
    it = iter(items[6:])
    return dict(zip(it, it))


PARSINGS = {'sale.order_create': _extract_key_value}


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
                parse_method = PARSINGS.get(action, parse_null)
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
