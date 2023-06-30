"""This module is responsible for parsing an SQL query string into its components
"""
import re
import sqlparse as sql
from sqlparse.sql import Parenthesis, Values

import case_id_assignment.utilities as util
from tqdm import tqdm
import pandas as pd
import numpy as np

SPLIT_REGEX = r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)"


def get_columns():
    return ['FileName',
            'BusinessActivity',
            'InstanceNumber',
            'sniff_time',
            'frame.number',
            'synthetic_sniff_time',
            'synthetic_sniff_time_str',
            'session_generalized',
            'HighestLayerProtocol',
            'MessageType_WithRole',
            'MessageType',
            'MessageAttributes',
            'query_type',
            'session_class',
            'filter_flag',
            'query',
            'tables',
            'event',
            'event_with_roles',
            'noise_event']


def filter_tokens(keys, token_cls):
    return [key for key in keys if isinstance(key, token_cls)]


def extract_function(keys):
    parenthesis = filter_tokens(keys, sql.sql.Parenthesis)
    identifier_list = [token for token in parenthesis[0] if isinstance(token, sql.sql.IdentifierList)]
    return identifier_list[0].value.split(',')


def extract_comparisons(identifier_list):
    return {clean(item.left.value): clean(item.right.value) for item in identifier_list if
            isinstance(item, sql.sql.Comparison)}


def replace_id_with_table_name(table_name, global_dict):
    try:
        id_value = global_dict['id']
        del global_dict['id']
        id_value: str = id_value.replace('(', '').replace(')', '')
        global_dict[f'{table_name}_id'] = id_value
    except KeyError:
        pass
    return global_dict


def extract_table_name(parsed):
    identifiers = [token for token in parsed[0] if isinstance(token, sql.sql.Identifier)]
    identifier = util.first_item(identifiers)
    table_name = clean(identifier.value)
    return table_name


def split_update_statement(parsed):
    items = [token for token in parsed[0] if
             isinstance(token, sql.sql.IdentifierList) or isinstance(token, sql.sql.Where)]
    global_dict = {}
    for item in items:
        global_dict.update(extract_comparisons(item))
    table_name = extract_table_name(parsed)
    global_dict = replace_id_with_table_name(table_name, global_dict)
    # identifier_list, where_clause = items
    # identifier_comparisons = extract_comparisons(identifier_list)
    # where_clause_comparisons = extract_comparisons(where_clause)
    # identifier_comparisons.update(where_clause_comparisons)
    return global_dict


def filter_items(tokens):
    return [str(token) for token in tokens if
            token.ttype != sql.tokens.Punctuation and token.is_whitespace is not True]


def get_clean_keys_and_values(keys, tokens, values):
    if any([isinstance(token, sql.sql.Function) for token in tokens]):
        keys, values = tokens
        strip_keys = extract_function(keys)
        parenthesis = filter_tokens(values, sql.sql.Parenthesis)
        identifier_list = filter_tokens([token for item in parenthesis for token in item], sql.sql.IdentifierList)
        strip_values = [item.value.split(',') for item in identifier_list]
        # first_column = [item[0] for item in strip_values]
        # second_column = [item[1] for item in strip_values]
        # reorder_values = [first_column, second_column]
        return strip_keys, strip_values[0]
    else:
        strip_keys = filter_items(keys[1])
        strip_values = re.split(SPLIT_REGEX, str(values[2])[1:-1])
        strip_values = [value.strip() for value in strip_values]
        return strip_keys, strip_values


def clean(value):
    """Cleans a string value removes apostrophe(s)"""
    return value.replace('"', '').replace("'", "").strip()


def split_insert_statement(parsed):
    tokens = [token for token in parsed[0] if
              isinstance(token, Parenthesis) or isinstance(token, Values) or isinstance(token, sql.sql.Function)]

    keys, values = tokens
    strip_keys, strip_values = get_clean_keys_and_values(keys, tokens, values)
    return {clean(k): clean(v) for k, v in zip(strip_keys, strip_values)}


def default_attributes(list_of_attributes, row):
    return {attribute: row[attribute] for attribute in list_of_attributes}


def parse_query(query_data):
    if not isinstance(query_data, str):
        return {}
    query_data = query_data.replace(r'\xa', ' ').strip()
    parsed = sql.parse(query_data)
    first_token = parsed[0][0]
    split_function = split_insert_statement if first_token.value == 'INSERT' else split_update_statement
    data = split_function(parsed)
    return data


def break_sql_query(list_of_attributes):
    def break_query(row):
        query_data = row['query']
        default_data = default_attributes(list_of_attributes, row)

        if query_data is None:
            return default_data
        try:
            data = parse_query(query_data)
        except ValueError as ex:
            print(f'Error parsing pnum:{row.name}')

        default_data.update(data)
        return default_data

    return break_query
