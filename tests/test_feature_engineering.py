"""This module tests the basic functionality of the feature engineering module"""
import os

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from sklearn.pipeline import Pipeline

import case_id_assignment.utilities as util
import case_id_assignment.feature_engineering as features_eng
from . import testutils as tu


def _feature_engineering_test_message(actual_columns: set, expected_columns: set) -> str:
    """Summary message for the generate feature test. prints which columns are missing and which are redundant"""
    missing = expected_columns - actual_columns
    redundant = actual_columns - expected_columns
    missing_message = f'The following columns are missing:{missing}\n' if missing else ''
    redundant_message = f'The following columns are redundant:{redundant}' if redundant else ''
    final_message = f'{missing_message}{redundant_message}' if missing_message or redundant_message else \
        'No columns are missing or redundant'
    return final_message


def generate_features_from_sql():
    """ Given a dataframe containing a column with an SQL query named query
        WHEN running the generate features function
        THEN a new dataframe with new columns from the query parameter will be generated
    """
    data_set_sample = tu.load_sample_data()
    data_set_sample_processed = features_eng.generate_features_from_sql(data_set=data_set_sample)
    actual_columns = set(data_set_sample_processed.columns)
    expected_columns = {'FileName', 'BusinessActivity', 'InstanceNumber', 'sniff_time', 'frame.number',
                        'synthetic_sniff_time', 'synthetic_sniff_time_str', 'session_generalized',
                        'HighestLayerProtocol', 'MessageType_WithRole', 'MessageType', 'MessageAttributes',
                        'query_type', 'session_class', 'filter_flag', 'query', 'tables', 'event', 'event_with_roles',
                        'noise_event', 'id', 'create_uid', 'create_date', 'write_uid', 'write_date', 'company_id',
                        'date_order', 'name', 'note', 'partner_id', 'res_id', 'partner_invoice_id',
                        'partner_shipping_id', 'picking_policy', 'pricelist_id', 'require_payment', 'require_signature',
                        'state', 'team_id', 'user_id', 'validity_date', 'warehouse_id', 'res_model',
                        'mail_followers_id', 'mail_message_subtype_id', 'currency_rate'}
    test_message = _feature_engineering_test_message(actual_columns, expected_columns)
    assert actual_columns == expected_columns, test_message


def test_generate_features_from_http():
    data_set_sample = tu.load_sample_data()
    data_set_sample_processed = features_eng.generate_features_from_http(data_set=data_set_sample)
    actual_columns = set(data_set_sample_processed.columns)
    expected_columns = {'FileName', 'BusinessActivity', 'InstanceNumber', 'sniff_time', 'frame.number',
                        'synthetic_sniff_time', 'synthetic_sniff_time_str', 'session_generalized',
                        'HighestLayerProtocol', 'MessageType_WithRole', 'MessageType', 'MessageAttributes',
                        'query_type', 'session_class', 'filter_flag', 'query', 'tables', 'event', 'event_with_roles',
                        'noise_event', 'request_method_call', 'starting_frame_number',
                        'file_data', 'activities', 'Unnamed: 0.3', 'activities_with_bp', 'frame_number',
                        'real_activity', 'real_activity_action', 'real_case_id', 'stream_index'}
    test_message = _feature_engineering_test_message(actual_columns, expected_columns)
    assert actual_columns == expected_columns, test_message


def test_feature_engineering():
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    pipeline = Pipeline(steps=[
        ('feature_eng', features_eng.EngineerFeatures())
    ])
    results = pipeline.fit_transform(sample_data)
    # util.save_data_set(data_set=results,data_folder='../data_for_tests',file_name='fe_res.csv')

    data = {
        'message_id_0': ['949753693051100', '574050965068213', '811388984381575', '066811463508817', '713444952148438'],
        'message_id_1': ['1608530433', '1608530433', '1608530795', '1608543481', '1608543481'],
        'message_id_2': ['504944801330566', '547146558761597', '007047414779663', '218040466308594', '232110261917114'],
        'message_id_3': ['', '', '152-purchase', '', ''],
        'message_id_4': ['@BPW10OD01', '@BPW10OD01', 'order@BPW10OD01', '@BPW10OD01', '@BPW10OD01']}
    indices = [15., 16., 17., 57., 58.]

    actual = results.loc[indices, list(data.keys())]
    expected = pd.DataFrame(data, index=indices)

    assert_frame_equal(actual, expected)
