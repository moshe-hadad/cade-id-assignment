"""This is the main module of the project. The main method contains all the steps of the project
1) Loading the data
2) Pre processing
3) Feature Engineering
4) Feature Selection
5) Data imputing
6) Clustering
7) Case id assignment
8) Evaluation
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import case_id_assignment.feature_engineering as features_eng
import case_id_assignment.feature_selection as selector
import case_id_assignment.clustering as clustering
import case_id_assignment.utilities as util
import case_id_assignment.imputing as imputer
import case_id_assignment.sqlutil as sql
import case_id_assignment.assignment as case_id_assigner
import case_id_assignment.evaluation as evaluation


def _sync_data_types(data_set: pd.DataFrame):
    dtypes = util.load_data_set(file_path='../../processed_data/dtypes_for_interleaved_df_imputed.csv')
    for column, dtype in dtypes.items():
        if column in data_set.columns:
            data_set[column] = data_set[column].astype(dtype)
    return data_set


def impute(interleaved_data_set_engineered: pd.DataFrame,
           save_results: bool = False, impute: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function perform the imputing stage on the given data sets. The basic method is hot deck method in which
    rows with similar values are imputed from one another. Or values are being completed by extracting them from other
    columns' values.

    :param interleaved_data_set_engineered: The interleaved data set
    :param save_results: If ture, the resulted data sets will be saved to the processed_data folder.
    :param impute: If True, the imputing stage will be performed on the data sets. If False the function will try to
    load and return the imputed data sets from the processed_data folder.
    :return: interleaved data set after imputing
    """
    processed_data_folder = '../../processed_data'
    interleaved_df_imputed_name = 'interleaved_df_imputed'
    if impute:
        missing_before = util.missing_data_percentage(interleaved_data_set_engineered)
        print(f'Missing data percentage before imputation:{missing_before}')

        columns = ['subject', 'origin', 'res_name', 'datas_fname']
        cleaner = {'purchase_order_id': [1., 2.],
                   'sale_order_id': [1., 2.],
                   'purchase_requisition_line_id': [1., 2.],
                   'purchase_requisition_id': [1., 2.],
                   'sale_order_line_id': [1.],
                   'account_invoice_id': [1.],
                   'stock_move_line_id': [1.]}

        impute_pipeline = Pipeline(steps=[
            ('filedata_imputer', imputer.ImputeFromFileData()),
            ('html_imputer', imputer.ImputeFromHtml()),
            ('po_imputer', imputer.ImputePO(columns)),
            ('res_imputer', imputer.ImputeFromRes()),
            ('requesr_method_imputer', imputer.ImputeFromRequestMethodCall()),
            ('name_column_imputer', imputer.ImputeFromNameColumn()),
            ('clean_values', features_eng.CleanValues(cleaner)),
            ('similar_columns_imputer', imputer.ImputeFromSimilarColumns()),
            ('stream_index_http_imputer', imputer.ImputeFromStreamIndexHTTP()),
            ('stream_index_imputer', imputer.ImputeFromStreamIndex(window_size=5))
        ])

        interleaved_df_imputed = impute_pipeline.fit_transform(interleaved_data_set_engineered)
        # interleaved_df_imputed = _sync_data_types(data_set=interleaved_df_imputed)

        missing_after = util.missing_data_percentage(interleaved_data_set_engineered)
        print(f'Missing data percentage after imputation:{missing_after}')

        if save_results:
            util.save_data_set(data_set=interleaved_df_imputed, data_folder='../../processed_data',
                               file_name=interleaved_df_imputed_name)
    else:
        interleaved_df_imputed = util.load_data_set(
            file_path=f'{processed_data_folder}/{interleaved_df_imputed_name}')

    return interleaved_df_imputed


def impute_based_on_row(data_set: pd.DataFrame, list_of_columns: list[str]):
    """This function performs the imputing based on columns (features) which are correlated with the instance number
    For each such column, we filter the data set based on its unique values.
    Then we complete all other values for all columns in the filtered data set
    The idea is that each value of a correlated feature, points to an instance id and thus all other values in the
    filtered data set, are also related to the same instance

    :param data_set: the data set to perform the imputation on
    :param list_of_columns: list of instance correlated values
    :return: imputed data set
    """
    impute_pipeline = Pipeline(steps=[
        ('row_imputer', imputer.ImputeFromSimilarRows(columns=list_of_columns,
                                                      excluded_columns=['real_case_id',
                                                                        'request_method_call',
                                                                        'starting_frame_number', 'file_data']))])

    imputed_data_set = impute_pipeline.fit_transform(X=data_set)
    return imputed_data_set


def engineer_features(isolated_df_processed: pd.DataFrame, interleaved_df_processed: pd.DataFrame,
                      save_results: bool = False, feature_engineering: bool = True) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """This function perform feature engineering on the given data sets
    The given data sets needs to be after the pre-processing stage, meaning the SQL queries and message attributes
    were broken to additional columns.

    :param isolated_df_processed: The isolated data set
    :param interleaved_df_processed: The interleaved data set
    :param save_results: If ture, the function will save the results into the processed_data folder
    :param feature_engineering: If True, the function will perform the feature_engineering stage. If False the function
    will try to load and return a previously featured engineered data set from the processed_data folder
    :return: isolated data set  and interleaved data set after feature engineering
    """
    cleaner = {'purchase_order_id': [1, 2],
               'sale_order_id': [1, 2],
               'purchase_requisition_line_id': [1, 2],
               'purchase_requisition_id': [1, 2],
               'sale_order_line_id': [1]}
    processed_data_folder = '../../processed_data'
    isolated_df_engineered_name = 'isolated_df_engineered'
    interleaved_df_engineered_name = 'interleaved_df_engineered'
    if feature_engineering:
        pipeline = Pipeline(steps=[
            ('feature_eng', features_eng.EngineerFeatures()),
            ('features_from_table', imputer.CreateFeaturesFromTableColumn()),
            ('clean_values', features_eng.CleanValues(cleaner))
        ])
        isolated_df_engineered = pipeline.fit_transform(isolated_df_processed)
        interleaved_df_engineered = pipeline.fit_transform(interleaved_df_processed)

        isolated_df_engineered = _str_to_nan(isolated_df_engineered)
        interleaved_df_engineered = _str_to_nan(interleaved_df_engineered)

        if save_results:
            util.save_data_set(data_set=isolated_df_engineered, data_folder='../../processed_data',
                               file_name=isolated_df_engineered_name)
            util.save_data_set(data_set=interleaved_df_engineered, data_folder='../../processed_data',
                               file_name=interleaved_df_engineered_name)
    else:
        isolated_df_engineered = util.load_data_set(file_path=f'{processed_data_folder}/{isolated_df_engineered_name}')
        interleaved_df_engineered = util.load_data_set(
            file_path=f'{processed_data_folder}/{interleaved_df_engineered_name}')

    return isolated_df_engineered, interleaved_df_engineered


def _str_to_nan(data_set):
    """Convert empty strings or NULL strings into np.nan """
    return data_set.replace('', np.nan).replace('NULL', np.nan).replace(' NULL',
                                                                        np.nan)


def pre_processing_data(isolated_data_set: pd.DataFrame, interleaved_data_set: pd.DataFrame, save_results: bool = False,
                        pre_process: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function breaks SQL queries and Message attributes to created additional columns.

    :param isolated_data_set: The isolated data set
    :param interleaved_data_set: he interleaved data set
    :param save_results:  If ture, the function will save the results into the processed_data folder
    :param pre_process: If True, the function will perform the pre-processing stage. If False the function
    will try to load and return a previously pre-processed data set from the processed_data folder
    :return: isolated data set  and interleaved data set after pre-processing
    """
    processed_data_folder = '../../processed_data'
    isolated_df_processed_name = 'isolated_df_processed'
    interleaved_df_processed_name = 'interleaved_df_processed'

    if pre_process:
        columns = sql.get_columns()
        print('Process isolated data set - parse SQL queries to features')
        isolated_df_processed = features_eng.generate_features_from_sql(data_set=isolated_data_set, columns=columns)
        print('Process interleaved data set - parse SQL queries to features')
        columns.extend(['real_activity', 'real_activity_action', 'real_case_id', 'activities_with_bp', 'stream_index'])
        interleaved_df_processed = features_eng.generate_features_from_sql(data_set=interleaved_data_set,
                                                                           columns=columns)

        print('Process isolated data set - HTTP attributes to features')
        isolated_df_processed = features_eng.generate_features_from_http(data_set=isolated_df_processed)
        print('Process interleaved data set - HTTP attributes to features')
        interleaved_df_processed = features_eng.generate_features_from_http(data_set=interleaved_df_processed)

        isolated_df_processed = _str_to_nan(isolated_df_processed)
        interleaved_df_processed = _str_to_nan(interleaved_df_processed)

        if save_results:
            util.save_data_set(data_set=isolated_df_processed, data_folder=processed_data_folder,
                               file_name=isolated_df_processed_name)
            util.save_data_set(data_set=interleaved_df_processed, data_folder=processed_data_folder,
                               file_name=interleaved_df_processed_name)
    else:
        isolated_df_processed = util.load_data_set(file_path=f'{processed_data_folder}/{isolated_df_processed_name}')
        interleaved_df_processed = util.load_data_set(
            file_path=f'{processed_data_folder}/{interleaved_df_processed_name}')

    return isolated_df_processed, interleaved_df_processed


PRE_PROCESS = False
FEATURE_ENGINEERING = False
IMPUTING = False

if __name__ == '__main__':
    # ----------------------------------------  Load data sets ------------------------------------------------------ #
    data_folder = '../../data'
    isolated_data_set = util.load_data_set(file_path=f'{data_folder}/ptp_isolated_data.csv')
    interleaved_data_set = util.load_data_set(file_path=f'{data_folder}/ptp_interleaved_data.csv')

    # ------------------------------------------ Pre-processing  ---------------------------------------------------- #
    isolated_df_processed, interleaved_df_processed = pre_processing_data(isolated_data_set, interleaved_data_set,
                                                                          save_results=True,
                                                                          pre_process=PRE_PROCESS)

    # ---------------------------------------  Feature Engineering -------------------------------------------------- #
    del isolated_df_processed['multi']
    del interleaved_df_processed['multi']
    # ---- Feature Engineering, manipulate features, change format, process value etc --- #
    isolated_df_engineered, interleaved_df_engineered = engineer_features(isolated_df_processed,
                                                                          interleaved_df_processed,
                                                                          save_results=True,
                                                                          feature_engineering=FEATURE_ENGINEERING)

    # ---------------------------------------- Feature Selection ------------------------------------------------------#
    # Selecting features based on uniquely covering instance number column
    list_of_features = selector.simple_correlation_selector(isolated_df_engineered, target_column='InstanceNumber',
                                                            threshold=0.90)

    print(f'Selected Features:{list_of_features}')

    # ---------------------------------- Impute Data, complete missing values ---------------------------------------- #
    del interleaved_df_engineered['MessageAttributes']
    interleaved_df_imputed = impute(interleaved_df_engineered, save_results=True, impute=IMPUTING)
    # Reload interleaved_df_imputed so dtypes will be synced
    interleaved_df_imputed = util.load_data_set(file_path='../../processed_data/interleaved_df_imputed.csv')
    filtered_df = interleaved_df_imputed[list_of_features]

    # ------------------------------------- Cluster values into groups ------------------------------------------------#
    clusters = clustering.greedy_modularity_communities(data_set=filtered_df)
    # clusters = clustering.girvan_newman(data_set=filtered_df)
    # clusters = clustering.louvain_communities(data_set=filtered_df)

    print(f'Number of clusters : {len(clusters)}')

    # ------------------------------------------- Assign case id ------------------------------------------------------#
    results_data_set = case_id_assigner.assign_case_id(data_set=interleaved_df_imputed, attributes=list_of_features,
                                                       clusters=clusters)
    results_data_set = case_id_assigner.assign_case_id_to_activity_action(data_set=interleaved_df_imputed)
    util.save_data_set(data_set=results_data_set, data_folder='../../processed_data', file_name='final_results.csv')

    # --------------------------------------- Evaluate the case id assignment------------------------------------------#
    rand_score, homogeneity, completeness, v_measure = evaluation.evaluate_case_id_accuracy(
        data_set=results_data_set)

    print(f'Rand Score : {rand_score}')
    print(f'Homogeneity : {homogeneity}')
    print(f'Completeness : {completeness}')
    print(f'V_measure : {v_measure}')
