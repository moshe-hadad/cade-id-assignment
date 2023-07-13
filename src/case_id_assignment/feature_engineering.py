"""This module holds logic for the feature engineering
It contains methods to break down columns to several columns, combined different columns into one, encoding string to
numerical values, converting and scaling columns
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm
from functools import reduce

import case_id_assignment.sqlutil as sql
import case_id_assignment.httputil as http


# class BreakDownSQLQueries()

def get_index(data_set, item):
    """Returns the index of the given item in the given dataframe
    The item is located using the values in its frame.number, BusinessActivity and InstanceNumber columns"""
    frame_number = item['frame.number']
    instance_number = item['InstanceNumber']
    activity = item['BusinessActivity']
    index = data_set[(data_set['frame.number'] == frame_number) & (data_set['InstanceNumber'] == instance_number) & (
            data_set['BusinessActivity'] == activity)].index
    return index


def generate_features_from_sql(data_set: pd.DataFrame, columns):
    """Converts the column query from the given dataframe, which containing an SQL query, into additional columns
    The column query contains an SQL query in a string format.
    A key value pair from the SQL query is converted to a column value pair.
    Rows which does not contain an SQL query or the SQL query does not contain a value for the key, will be assigned
    an empty value.

    For a data frame of this form :
                id  | frame.number  | protocol  | query
                1   | 1325          | HTTP      | NONE
                2   | 8562          | PGSQL     | INSERT INTO "res_users_log" ("create_uid", "write_uid") VALUES
                                                                              (2, 2) RETURNING id
                3   | 2345          | HTTP      |
                4   | 3654          | PGSQL     | UPDATE "sale_order" SET "currency_rate"='1.000000',"write_uid"=1,
                                                  WHERE id IN (94)
    The result is a sparse matrix (for brevity, the column query was omitted)
                id  | frame.number  | protocol  | create_uid    | write_uid     | id    | currency_rate
                1   | 1325          | HTTP      | NONE          | NONE          | NONE  | NONE
                2   | 8562          | PGSQL     | 2             | 2             | NONE  | NONE
                3   | 2345          | HTTP      | NONE          | NONE          | NONE  | NONE
                4   | 3654          | PGSQL     | NONE          | 1             | 94    |'1.000000'


        :param data_set:  a dataframe which contains a column named query
        :return: a data frame containing the additional columns added
        """
    tqdm.pandas(desc='Convert SQL Queries into additional columns')
    parsed_data = data_set.progress_apply(sql.break_sql_query(columns), axis=1)
    # merged_dict = merge_data(parsed_data, columns)
    print('finished processing the data, create data frames')
    # complete_data_set = pd.DataFrame(merged_dict)
    # data_frames = [pd.DataFrame(item, index=['pnumber']) for item in parsed_data if item]
    data_frames = [pd.DataFrame(item, index=get_index(data_set, item)) for item in parsed_data if item]
    print('merge data frames to one global data frame')
    global_df = reduce(lambda df1, df2: pd.concat([df1, df2]), data_frames)
    return global_df


def generate_features_from_http(data_set: pd.DataFrame) -> pd.DataFrame:
    """Converts the columns MessageAttributes from the given dataframe into additional columns.
    The column MessageAttributes contains an HTTP request in a JSON format.
    Potential attributes, such as request method call, or HTTP request attributes, are extracted from this JSON as
    additional columns.

    :param data_set: a dataframe which contains a column named MessageAttributes
    :return: a data frame containing the additional columns added
    """
    tqdm.pandas(desc='Convert HTTP attributes into additional columns')
    results = data_set.apply(http.parse_method_and_file_data, axis=1)
    data_set[['request_method_call', 'starting_frame_number', 'file_data']] = pd.DataFrame(results.tolist())
    return data_set


class EngineerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.message_id_columns = None

    def fit(self, X, y=None):
        new_ds = X.dropna(how='all')
        self.message_id_columns = new_ds['message_id'].str.replace('<|>', '', regex=True).str.replace(
            '-hr', '', regex=False).str.replace(
            'private|message-notify|-openerp-', '.', regex=True).str.split('.', expand=True).add_prefix('message_id_')
        return self

    def transform(self, X, y=None):
        X = X.join(self.message_id_columns)
        return X
