import case_id_assignment.utilities as util
import case_id_assignment.feature_selection as selector
import case_id_assignment.clustering as clust
from . import testutils as tu


# def test_cluster():
#     imputed_data_set = tu.load_sample_data(file_name='sample_for_clustering.csv')
#     features = selector.simple_correlation_selector(data_set=imputed_data_set, target_column='InstanceNumber',
#                                                     threshold=0.95)
#     print(features)

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
def test_greedy_modularity_communities():
    imputed_data_set = util.load_data_set('C:/Research/cade-id-assignment/processed_data/interleaved_df_imputed.csv')
    features = ['checksum', 'message_id_3', 'sale_order_line_id', 'mail_mail_id', 'stock_move_id', 'mail_message_id',
                'message_id_1', 'store_fname', 'requisition_id', 'account_invoice_id', 'purchase_requisition_id',
                'account_move_id', 'datas_fname', 'message_id_0', 'stock_move_line_id', 'mail_followers_id', 'origin',
                'stock_picking_id', 'purchase_requisition_line_id', 'ir_attachment_id', 'parent_id', 'body_html',
                'account_move_line_id', 'invoice_id', 'account_invoice_line_id', 'ref', 'subject', 'attachment_id',
                'message_id', 'record_name', 'reference', 'purchase_order_line_id', 'res_name', 'move_name',
                'purchase_line_id', 'sale_order_id', 'group_id', 'picking_id', 'number', 'purchase_order_id']
    filtered = imputed_data_set[features]
    filtered = filtered[:100]
    clusters = clust.greedy_modularity_communities(data_set=filtered, save_to_file=False)
