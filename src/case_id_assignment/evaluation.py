import numpy as np
from sklearn import metrics
import case_id_assignment.utilities as util


def _to_int(value):
    if value == 'set()':
        return 0
    return int(value.replace('{', '').replace('}', ''))


def _non_null_value(data_set, column):
    values = data_set[data_set[[column]].notnull().all(1)][column].to_list()
    return [_to_int(value) for value in values]


def evaluate_case_id_accuracy(data_set):
    real_case_id_column = 'real_case_id'
    predicted_case_id_column = 'CaseIDVoting'
    case_id_mapping_column = 'CaseIDMapping'
    data_set[case_id_mapping_column] = data_set[real_case_id_column].map(util.case_id_mapping())
    # case_id_data_set = data_set[data_set[real_case_id_column] != '']

    labels_true = _non_null_value(data_set=data_set, column=case_id_mapping_column)
    labels_pred = _non_null_value(data_set=data_set, column=predicted_case_id_column)

    result = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return result
    #
    # start_case_id_accuracy, end_case_id_accuracy = report_clustering_accuracy(case_id_data_set,
    #                                                                           truth_column=case_id_mapping_column,
    #                                                                           predicted_column=predicted_case_id_column)
    # return start_case_id_accuracy, end_case_id_accuracy


def report_clustering_accuracy(case_id_data_set, truth_column, predicted_column):
    start_case_id_data_set = case_id_data_set[case_id_data_set['real_single_activity_action'] == 'Activity Start']
    start_case_id_accuracy = _calculate_accuracy(start_case_id_data_set, true_column=truth_column,
                                                 predicted_column=predicted_column)
    end_case_id_data_set = case_id_data_set[case_id_data_set['real_single_activity_action'] == 'Activity End']
    end_case_id_accuracy = _calculate_accuracy(end_case_id_data_set, true_column=truth_column,
                                               predicted_column=predicted_column)

    return start_case_id_accuracy, end_case_id_accuracy


def _calculate_accuracy(data_set, true_column, predicted_column):
    equality_test = np.where(data_set[true_column] == data_set[predicted_column], 1, 0)
    accuracy = sum(equality_test) / len(equality_test)
    return accuracy
