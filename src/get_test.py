import operator
from collections import defaultdict
from pprint import pprint

import numpy as np

import testutils as tu


def _extract_key_value(items):
    it = iter(items[6:])
    return dict(zip(it, it))


ACTION_NAME = operator.itemgetter(0, 4, 5)


def gen_test_from_file_data(test_data):
    def parse_file_data(row):
        file_data = row['file_data']
        if not isinstance(file_data, str) and np.isnan(file_data):
            return np.NAN
        file_data = eval(file_data)
        if len(file_data) < 5:
            return np.NAN
        key = '_'.join(ACTION_NAME(file_data))
        dict_data = _extract_key_value(file_data)
        # dict_data['index'] = row.name
        test_data[key].append((row.name, dict_data))

    return parse_file_data


def get_data(values):
    indices = []
    final_dict = defaultdict(list)
    for item in values:
        index, data_dict = item
        indices.append(index)
        for key, value in data_dict.items():
            final_dict[key].append(value)
    return indices, final_dict


if __name__ == '__main__':
    sample_data = tu.load_sample_data(file_name='sample_for_imputing.csv')
    test_data = defaultdict(list)
    sample_data.apply(gen_test_from_file_data(test_data), axis=1)
    for key, values in test_data.items():
        test_name = key.replace('execute_kw_', '').replace('.','_')
        indices, expected = get_data(values)
        text = f"""
        def test_impute_{test_name}():
            results = _input_sample_data()
            data = {dict(expected)}
            indices = {indices}
        
            actual = results.loc[indices, list(data.keys())]
            expected = pd.DataFrame(data, index=indices)
        
            assert_frame_equal(actual, expected)
        """
        print(text)
        print('\n')

    # pprint(dict(test_data))
