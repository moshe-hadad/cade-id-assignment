import case_id_assignment.utilities as util
import case_id_assignment.evaluation as evaluation


def test_evaluate_case_id_accuracy():
    benchmark_path = '../processed_data/benchmark.csv'
    benchmark_df = util.load_data_set(file_path=benchmark_path)

    benchmark_eval = evaluation.evaluate_case_id_accuracy(data_set=benchmark_df)
    assert benchmark_eval == 0.45331918313514974
