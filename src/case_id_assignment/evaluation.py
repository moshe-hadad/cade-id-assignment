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

    filtered = data_set[data_set['real_activity_action'] != 'NoAction']
    labels_true = _non_null_value(data_set=filtered, column=case_id_mapping_column)
    labels_pred = _non_null_value(data_set=filtered, column=predicted_case_id_column)

    rand_score = metrics.rand_score(labels_true, labels_pred)
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

    return rand_score, homogeneity, completeness, v_measure
