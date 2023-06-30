"""This module holds logic for the feature engineering
It contains methods to break down columns to several columns, combined different columns into one, encoding string to
numerical values, converting and scaling columns
"""

from tqdm.auto import tqdm
import pandas as pd
import sklearn.pipeline as pipe
import case_id_assignment.sqlutil as sql
from functools import reduce


# class BreakDownSQLQueries()

def get_index(data_set, item):
    frame_number = item['frame.number']
    instance_number = item['InstanceNumber']
    activity = item['BusinessActivity']
    index = data_set[(data_set['frame.number'] == frame_number) & (data_set['InstanceNumber'] == instance_number) & (
            data_set['BusinessActivity'] == activity)].index
    return index


def convert_sql_to_features(data_set: pd.DataFrame) -> pd.DataFrame:
    """Responsible for converting a column containing an SQL string into additional columns
    A key value pair from the SQL query is converted to a column value pair
    Rows which does not contain an SQL query or the SQL query does not contain a value for the key, will be assigned an
    empty value

    For a data frame of this form :
                id  | frame.number  | protocol  | Query
                1   | 1325          | HTTP      | NONE
                2   | 8562          | PGSQL     | INSERT INTO "res_users_log" ("create_uid", "write_uid") VALUES
                                                                              (2, 2) RETURNING id
                3   | 2345          | HTTP      |
                4   | 3654          | PGSQL     | UPDATE "sale_order" SET "currency_rate"='1.000000',"write_uid"=1,
                                                  WHERE id IN (94)
    The result is a sparse matrix
                id  | frame.number  | protocol  | create_uid    | write_uid     | id    | currency_rate
                1   | 1325          | HTTP      | NONE          | NONE          | NONE  | NONE
                2   | 8562          | PGSQL     | 2             | 2             | NONE  | NONE
                3   | 2345          | HTTP      | NONE          | NONE          | NONE  | NONE
                4   | 3654          | PGSQL     | NONE          | 1             | 94    |'1.000000'


    :param data_set: pd.DataFrame data frame to work on
    :return: data frame with the additional columns added
    """
    tqdm.pandas(desc='Convert SQL Queries into additional columns')
    columns = sql.get_columns()
    parsed_data = data_set.progress_apply(sql.break_sql_query(columns), axis=1)
    # merged_dict = merge_data(parsed_data, columns)
    print('finished processing the data, create data frames')
    # complete_data_set = pd.DataFrame(merged_dict)
    # data_frames = [pd.DataFrame(item, index=['pnumber']) for item in parsed_data if item]
    data_frames = [pd.DataFrame(item, index=get_index(data_set, item)) for item in parsed_data if item]
    print('merge data frames to one global data frame')
    global_df = reduce(lambda df1, df2: pd.concat([df1, df2]), data_frames)
    return global_df


def generate_features(data_set: pd.DataFrame):
    # transformation_pipeline = pipe.Pipeline()
    data_set = convert_sql_to_features(data_set=data_set)

    return data_set
