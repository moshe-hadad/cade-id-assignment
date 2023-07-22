import pandas as pd

import case_id_assignment.assignment as assign


def test__enumerate_clusters():
    clusters = [['2022-08-03 10:16:10'],
                ['INV/2022/0143/46', 269.0, 542.0],
                ['6253'],
                ['2022-08-03 05:37:50.209211', 6257.0, 6267.0, '21028', 'PO00999', 999.0, '246'],
                ['6173'],
                ['6247'],
                ['2022-08-03 05:37:31.278157']]
    clusters_enum: dict[object, list] = assign._enumerate_clusters(clusters=clusters)

    assert clusters_enum['2022-08-03 10:16:10'].pop() == 0
    assert clusters_enum['INV/2022/0143/46'].pop() == 1
    assert clusters_enum[542.0].pop() == 1
    assert clusters_enum['6253'].pop() == 2
    assert clusters_enum['6173'].pop() == 4
    assert clusters_enum['6247'].pop() == 5
    assert clusters_enum['2022-08-03 05:37:31.278157'].pop() == 6


def test_assign_case_id():
    clusters = [[399, 980, 400, 981, 402, 978],
                [401, 979, 983, 982, 403, 984],
                [404, 985, 405, 986]
                ]

    data_set = pd.DataFrame({
        'sale_order_line_id': [399, 404, 400, 399, 401, 402, 401, 399, 403, 400, 403, 403, 405],
        'purchase_order_id': [980, 986, 981, 978, 979, 981, 983, 980, 982, 981, 984, 983, 985]
    })
    list_of_features = ['sale_order_line_id', 'purchase_order_id']
    results_data_set = assign.assign_case_id(data_set=data_set, attributes=list_of_features,
                                             clusters=clusters)
    assert 'case_id' in results_data_set.columns
    assert results_data_set['case_id'][0] == '{0}'
    assert results_data_set['case_id'][1] == '{2}'
    assert results_data_set['case_id'][2] == '{0}'
    assert results_data_set['case_id'][3] == '{0}'
    assert results_data_set['case_id'][4] == '{1}'


def test_vote_case_id():
    data_set = pd.DataFrame({
        'sale_order_line_id': [399, 404, 400, 399, 401, 402, 401, 399, 403, 400, 403, 403, 405],
        'purchase_order_id': [980, 986, 981, 978, 979, 981, 983, 980, 982, 981, 984, 983, 985],
        'real_activity_action': ['Activity Start', 'NoAction', 'Activity Start', 'NoAction', 'NoAction', 'Activity End',
                                 'NoAction', 'NoAction', 'Activity End', 'Activity Start', 'NoAction', 'NoAction',
                                 'Activity End'],
        'case_id': ['set()', '{1}', '{2,1}', '{2}', 'set()', 'set()', 'set()', '{3}', 'set()', 'set()', '{4,5}', '{5}',
                    'set()']
    })
    results_data_set = assign.assign_case_id_to_activity_action(data_set=data_set)
    assert 'CaseIDVoting' in results_data_set.columns
    assert results_data_set['CaseIDVoting'][0] == '{1}'
    assert results_data_set['CaseIDVoting'][2] == '{1}'
    assert results_data_set['CaseIDVoting'][5] == '{2}'
    assert results_data_set['CaseIDVoting'][8] == '{3}'
    assert results_data_set['CaseIDVoting'][9] == '{4}'
    assert results_data_set['CaseIDVoting'][12] == '{5}'
