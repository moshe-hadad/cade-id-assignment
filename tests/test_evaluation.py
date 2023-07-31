import case_id_assignment.utilities as util
import case_id_assignment.evaluation as evaluation


def test_evaluate_case_id_accuracy():
    benchmark_path = '../processed_data/benchmark.csv'
    benchmark_df = util.load_data_set(file_path=benchmark_path)

    rand_score, homogeneity, completeness, v_measure = evaluation.evaluate_case_id_accuracy(
        data_set=benchmark_df)

    print(f'Rand Score : {rand_score}')
    print(f'Homogeneity : {homogeneity}')
    print(f'Completeness : {completeness}')
    print(f'V_measure : {v_measure}')
    assert rand_score == 0.8009893455098934
    assert homogeneity == 0.44282849670130947
    assert completeness == 0.46431898304109215
    assert v_measure == 0.45331918313514974
