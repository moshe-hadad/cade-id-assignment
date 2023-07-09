import case_id_assignment.utilities as util
import case_id_assignment.feature_selection as selector


def test_simple_correlation_selector():
    sample_data = util.load_data_set(file_path='../data_for_tests/sample_for_correlation.csv')
    features = selector.simple_correlation_selector(data_set=sample_data, target_column='InstanceNumber',
                                                    threshold=0.95)

    actual = features
    expected = ['res_id', 'mail_followers_id', 'mail_message_id', 'order_id', 'sale_order_id', 'sale_order_line_id']
    assert actual == expected

# def test_data_sets():
#     actual = util.load_data_set('../processed_data/isolated_df_imputed.csv')
#     expected = util.load_data_set('../processed_data/order_to_cash_extended_features_benchmark.csv')
#     expected = set([item.strip() for item in expected.columns])
#     actual = set([item.strip() for item in actual.columns])
#     actual.remove('Unnamed: 0')
#     actual.remove('Unnamed: 0.1')
#     expected.remove('Unnamed: 0')
#
#     assert actual == expected

# def test_create_sample_for_correlation():
#     full_data = util.load_data_set('../processed_data/isolated_df_imputed.csv')
#     sample_data = full_data[:200]
#     util.save_data_set(sample_data, data_folder='../data_for_tests', file_name='sample_for_correlation.csv')
