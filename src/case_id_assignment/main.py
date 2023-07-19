import os.path

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


def impute(isolated_data_set_engineered: pd.DataFrame, interleaved_data_set_engineered: pd.DataFrame,
           save_results: bool = False, impute: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function perform the imputing stage on the given data sets. The basic method is hot deck method in which
    rows with similar values are imputed from one another. Or values are being completed by extracting them from other
    columns' values.

    :param isolated_data_set_engineered: The isolated data set
    :param interleaved_data_set_engineered: The interleaved data set
    :param save_results: If ture, the resulted data sets will be saved to the processed_data folder.
    :param impute: If True, the imputing stage will be performed on the data sets. If False the function will try to
    load and return the imputed data sets from the processed_data folder.
    :return:
    """
    processed_data_folder = '../../processed_data'
    isolated_df_imputed_name = 'isolated_df_imputed'
    interleaved_df_imputed_name = 'interleaved_df_imputed'
    if impute:
        columns = ['subject', 'origin', 'res_name', 'datas_fname']
        impute_pipeline = Pipeline(steps=[
            ('tableimputer', imputer.ImputeFromTable()),
            ('filedata_imputer', imputer.ImputeFromFileData()),
            ('html_imputer', imputer.ImputeFromHtml()),
            ('po_imputer', imputer.ImputePO(columns)),
            ('res_imputer', imputer.ImputeFromRes())
        ])
        isolated_df_imputed = impute_pipeline.fit_transform(isolated_data_set_engineered)
        interleaved_df_imputed = impute_pipeline.fit_transform(interleaved_data_set_engineered)
        if save_results:
            util.save_data_set(data_set=isolated_df_imputed, data_folder='../../processed_data',
                               file_name=isolated_df_imputed_name)
            util.save_data_set(data_set=interleaved_df_imputed, data_folder='../../processed_data',
                               file_name=interleaved_df_imputed_name)
    else:
        isolated_df_imputed = util.load_data_set(file_path=f'{processed_data_folder}/{isolated_df_imputed_name}')
        interleaved_df_imputed = util.load_data_set(
            file_path=f'{processed_data_folder}/{interleaved_df_imputed_name}')

    return isolated_df_imputed, interleaved_df_imputed


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
    processed_data_folder = '../../processed_data'
    isolated_df_engineered_name = 'isolated_df_engineered'
    interleaved_df_engineered_name = 'interleaved_df_engineered'
    if feature_engineering:
        pipeline = Pipeline(steps=[
            ('feature_eng', features_eng.EngineerFeatures())
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
    """Conver empty strings or NULL strings into np.nan"""
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
    # ---------------  Load data sets ---------------- #
    data_folder = '../../data'
    isolated_data_set = util.load_data_set(file_path=f'{data_folder}/ptp_isolated_data.csv')
    interleaved_data_set = util.load_data_set(file_path=f'{data_folder}/ptp_interleaved_data.csv')

    # ---- Pre-process the data by breaking down SQL queries and message attributes to add columns --- #
    isolated_df_processed, interleaved_df_processed = pre_processing_data(isolated_data_set, interleaved_data_set,
                                                                          save_results=True,
                                                                          pre_process=PRE_PROCESS)

    # ---- Feature Engineering, manipulate features, change format, process value etc --- #
    isolated_df_engineered, interleaved_df_engineered = engineer_features(isolated_df_processed,
                                                                          interleaved_df_processed,
                                                                          save_results=True,
                                                                          feature_engineering=FEATURE_ENGINEERING)

    # ---- Impute Data, complete missing values --- #
    isolated_df_imputed, interleaved_df_imputed = impute(isolated_df_engineered, interleaved_df_engineered,
                                                         save_results=True,
                                                         impute=IMPUTING)

    # Selecting features based on correlation
    list_of_features = selector.simple_correlation_selector(isolated_df_imputed, target_column='InstanceNumber',
                                                            threshold=0.90)

    # shared_columns = list(set(isolated_df_engineered.columns).intersection(set(list_of_features)))
    # isolated_missing_percentage_before = util.missing_data_percentage(isolated_df_engineered[shared_columns])
    # interleaved_missing_percentage_before = util.missing_data_percentage(interleaved_df_engineered[shared_columns])
    #
    # isolated_missing_percentage_after = util.missing_data_percentage(isolated_df_imputed[shared_columns])
    # interleaved_missing_percentage_after = util.missing_data_percentage(interleaved_df_imputed[shared_columns])
    #
    # print('*** Missing percentage before and after ***')
    # print(f'Isolated :before:{isolated_missing_percentage_before}, after:{isolated_missing_percentage_after}')
    # print(
    #     f'Interleaved :before:{interleaved_missing_percentage_before}, after:{interleaved_missing_percentage_after}')
    # print('*' * 50)

    print(list_of_features)
    # # Impute data on the interleaved data set

    filtered_df = interleaved_df_imputed[list_of_features]
    # Cluster values into groups
    clusters = clustering.cluster(data_set=filtered_df)
    print(clusters)
    print(len(clusters))

    # # Assign case id
    # # todo work on the assigner module
    # results_data_set = case_id_assigner.assigne_case_id(data_set=interleaved_df_processed, clusters=clusters)
    #
    # # Evaluate the case id assignment
    # # todo work on the evaluation
    # evaluation.print_evaluation(data_set=results_data_set)
