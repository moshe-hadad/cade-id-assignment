from collections import defaultdict

import pandas as pd
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
    tqdm.pandas(desc='Assign case id to packets')
    case_id_column = data_set.progress_apply(case_id_assignment(attributes, clusters_enum), axis=1)
    data_set['case_id'] = case_id_column
    return data_set
