import case_id_assignment.imputing as imp
from . import testutils as tu


def test_impute_from_message_attributes():
    sample_data = tu.load_sample_data()
    imputed_data = imp.impute_from_message_attributes(data_set=sample_data)
