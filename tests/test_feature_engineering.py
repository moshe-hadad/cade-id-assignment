"""This module tests the basic functionality of the feature engineering module"""
import os
import case_id_assignment.utilities as util
import case_id_assignment.feature_engineering as features_eng


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
    data_set_sample = load_sample_data()
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


def load_sample_data():
    data_folder = '../data_for_tests'
    data_sample_path = os.path.join(data_folder, 'data_sample.csv')
    data_set_sample = util.load_data_set(file_path=data_sample_path)
    return data_set_sample


def test_generate_features_from_http():
    data_set_sample = load_sample_data()
    data_set_sample_processed = features_eng.generate_features_from_http(data_set=data_set_sample)
    actual_columns = set(data_set_sample_processed.columns)
    expected_columns = {'FileName', 'BusinessActivity', 'InstanceNumber', 'sniff_time', 'frame.number',
                        'synthetic_sniff_time', 'synthetic_sniff_time_str', 'session_generalized',
                        'HighestLayerProtocol', 'MessageType_WithRole', 'MessageType', 'MessageAttributes',
                        'query_type', 'session_class', 'filter_flag', 'query', 'tables', 'event', 'event_with_roles',
                        'noise_event', 'Unnamed: 0.1', 'Unnamed: 0', 'request_method_call', 'starting_frame_number',
                        'file_data', }
    test_message = _feature_engineering_test_message(actual_columns, expected_columns)
    assert actual_columns == expected_columns, test_message
