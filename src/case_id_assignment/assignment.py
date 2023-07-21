from collections import defaultdict

import pandas as pd
import case_id_assignment.utilities as util
from tqdm.auto import tqdm


def _enumerate_clusters(clusters: list[list[object]]) -> dict[object, int]:
    enumeration = defaultdict(list)
    for enum, cluster in tqdm(enumerate(clusters), desc=''):
        for value in cluster:
            enumeration[value].append(enum)
    return dict(enumeration)


def case_id_assignment(attributes, clusters_enum):
    def assign(row):
        values = row[attributes]
        case_ids = {assignment for value in values for assignment in clusters_enum.get(value, '')}

        return str(case_ids)

    return assign


def assign_case_id(data_set: pd.DataFrame, attributes: list[str], clusters: list[list[object]]) -> pd.DataFrame:
    """Assign a case id to rows (packets) based on their attributes values.
    Iterate over the rows in data set, for each row, extract the values for the attributes.
    Identify a case id number based on the attributes values from the clusters
    Assign the case id number to the row

    :param data_set: The data set to assign case id on
    :param attributes: A list of attributes to extract from a row
    :param clusters:  A list of values' clusters
    :return: The data set with an additional column named 'CaseIDVoting' containing the case id assignment
    """
    clusters_enum = _enumerate_clusters(clusters)
    tqdm.pandas(desc='Assign case id to packets containing attributes values')
    case_id_column = data_set.progress_apply(case_id_assignment(attributes, clusters_enum), axis=1)
    data_set['case_id'] = case_id_column
    return data_set


def _closest_case_id_from_forward(data_set, index):
    filtered = data_set[index:]
    case_id_index = filtered[filtered['case_id'].notnull()].first_valid_index()
    case_id = _get_case_id_by_index(data_set=filtered, case_id_index=case_id_index)
    return case_id


def _closest_case_id_from_backwards(data_set, index):
    filtered = data_set[:index]
    case_id_index = filtered[filtered['case_id'].notnull()].last_valid_index()
    case_id = _get_case_id_by_index(data_set=filtered, case_id_index=case_id_index)
    return case_id


def _get_case_id_by_index(data_set, case_id_index):
    case_ids = data_set['case_id'].iloc[case_id_index]
    case_ids = eval(case_ids)
    case_id = util.first_item(list(case_ids))
    return case_id


def closest_case_id(data_set):
    def assign(row):
        activity_action = row['real_activity_action']
        case_ids = row['case_id']
        if activity_action == 'NoAction':
            return case_ids

        # If activity action is Activity Start or Activity End
        case_ids = eval(case_ids)
        if case_ids:  # if it's already assigned with a case id, if there is more than one value, we return the first
            case_id = util.first_item(case_ids) if len(case_ids) > 1 else case_ids
            return f'{case_id}'

        # If activity action is not assigned with a case id, borrow one from the closes case id
        if activity_action == 'Activity Start':
            case_id = _closest_case_id_from_forward(data_set, row.name)
        else:
            case_id = _closest_case_id_from_backwards(data_set, row.name)

        return f'{case_id}'

    return assign


def assign_case_id_to_activity_action(data_set: pd.DataFrame) -> pd.DataFrame:
    """Finds the location of all activity actions (start and end).
    For each activity action, assign the closest case id. For activity start, the closest looking forward
    For activity end, the closes looking backwards.
    If the closes case id has more than one value, choose the first one

    :param data_set: A data set with the following columns:  case_id columns containing values of case id e.g. {1} or
    {1,5}. activity_action containing activity action values e.g. Activity Start, NoAction and Activity End
    :return: A data set containing column "CaseIdVoting" with the case id values e.g. {1}
    """
    tqdm.pandas(desc="Assign the closest case id to activity actions")
    case_id_voting = data_set.progress_apply(closest_case_id(data_set), axis=1)
    data_set['CaseIDVoting'] = case_id_voting
    return data_set
