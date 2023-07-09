import os.path

from sklearn.pipeline import Pipeline

import case_id_assignment.feature_engineering as features_eng
import case_id_assignment.feature_selection as selector
import case_id_assignment.clustering as clustering
import case_id_assignment.utilities as util
import case_id_assignment.imputing as imputer
import case_id_assignment.assignment as case_id_assigner
import case_id_assignment.evaluation as evaluation

if __name__ == '__main__':
    data_folder = '../../data'

    # isolated_file_path = os.path.join(data_folder, 'ptp_isolated_data.csv')
    # interleaved_file_path = os.path.join(data_folder, 'ptp_interleaved_data.csv')
    #
    # isolated_data_set = util.load_data_set(file_path=isolated_file_path)
    # interleaved_data_set = util.load_data_set(file_path=interleaved_file_path)
    #
    # # Preprocess data, generate features
    # # todo work on the feature engineering module
    # print('Process isolated data set - parse SQL queries to features')
    # isolated_df_processed = features_eng.generate_features_from_sql(data_set=isolated_data_set)
    # # util.save_data_set(data_set=isolated_df_processed, data_folder=data_folder, name='isolated_df_processed')
    # print('Process interleaved data set - parse SQL queries to features')
    # interleaved_df_processed = features_eng.generate_features_from_sql(data_set=interleaved_data_set)
    # # util.save_data_set(data_set=isolated_df_processed, data_folder=data_folder, name='interleaved_df_processed')
    #
    # processed_data_folder = '../../processed_data'
    # print('Process isolated data set - HTTP attributes to features')
    # isolated_df_processed = features_eng.generate_features_from_http(data_set=isolated_df_processed)
    # util.save_data_set(data_set=isolated_df_processed, data_folder=processed_data_folder, name='isolated_df_processed')
    # print('Process interleaved data set - HTTP attributes to features')
    # interleaved_df_processed = features_eng.generate_features_from_http(data_set=interleaved_df_processed)
    # util.save_data_set(data_set=isolated_df_processed, data_folder=processed_data_folder,
    #                    name='interleaved_df_processed')

    isolated_df_processed = util.load_data_set(
        file_path=os.path.join('../../processed_data/', 'isolated_df_imputed.csv'))
    interleaved_df_processed = util.load_data_set(
        file_path=os.path.join('../../processed_data', 'interleaved_df_processed.csv'))
    # impute features values from MessageAttributes
    pipeline = Pipeline(steps=[
        ('inputter', imputer.ImputeFromFileData()),
        ('tableimputer', imputer.ImputeFromTable())
    ])

    # isolated_df_imputed = results = pipeline.fit_transform(isolated_df_processed)
    # interleaved_df_imputed = results = pipeline.fit_transform(interleaved_df_processed)
    #
    # util.save_data_set(data_set=isolated_df_imputed, data_folder='../../processed_data',
    #                    file_name='isolated_df_imputed.csv')
    # util.save_data_set(data_set=interleaved_df_imputed, data_folder='../../processed_data',
    #                    file_name='interleaved_df_imputed.csv')

    # Selecting features based on correlation
    list_of_features = selector.simple_correlation_selector(isolated_df_processed, target_column='InstanceNumber',
                                                            threshold=0.95)
    print(list_of_features)

    # # Impute data on the interleaved data set
    # # todo work on the imputing module
    # interleaved_df_processed = imputer.impute(data_set=interleaved_df_processed, method='')
    #
    # # Cluster values into groups
    # # todo work on the clustering module
    # clusters = clustering.cluster(file_path=interleaved_df_processed, clustering_method='')
    #
    # # Assign case id
    # # todo work on the assigner module
    # results_data_set = case_id_assigner.assigne_case_id(data_set=interleaved_df_processed, clusters=clusters)
    #
    # # Evaluate the case id assignment
    # # todo work on the evaluation
    # evaluation.print_evaluation(data_set=results_data_set)
