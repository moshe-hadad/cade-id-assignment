import case_id_assignment.utilities as util
import case_id_assignment.feature_selection as selector
from . import testutils as tu


def test_cluster():
    imputed_data_set = tu.load_sample_data(file_name='sample_for_clustering.csv')
    features = selector.simple_correlation_selector(data_set=imputed_data_set, target_column='InstanceNumber',
                                                    threshold=0.95)
    print(features)

    # full_data = util.load_data_set('../processed_data/interleaved_df_imputed.csv')
    # sample_data = full_data[:100]
    # util.save_data_set(sample_data, data_folder='../data_for_tests', file_name='sample_for_clustering.csv')

# def test_fix_file():
#     df = util.load_data_set('../data/ptp_interleaved_data.csv')
#     columns = list(df.columns)
#     columns.remove('Unnamed: 0.2')
#     columns.remove('Unnamed: 0.1')
#     columns.remove('Unnamed: 0')
#
#     new_df = df[columns]
#     util.save_data_set(data_set=new_df,data_folder='../data', file_name='ptp_interleaved_data.csv')