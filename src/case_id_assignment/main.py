import case_id_assignment.feature_engineering as engineer
import case_id_assignment.feature_selection as selector
import case_id_assignment.clustering as clustering
import case_id_assignment.utilities as util
import case_id_assignment.imputing as imputer
import case_id_assignment.assignment as case_id_assigner
import case_id_assignment.evaluation as evaluation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Generate features
    isolated_file_path = '../../data/ptp_isolated_data.csv'
    interleaved_file_path = '../../data/ptp_interleaved_data.csv'
    # todo commit the code to github
    # todo get the clean interleaved data set (no break down of feature)
    # todo work on the util module to load data set
    isolated_data_set = util.load_data_set(file_path=isolated_file_path)
    interleaved_data_set = util.load_data_set(file_path=interleaved_file_path)

    # Preprocess data, generate features
    # todo work on the feature engineering module
    processed_isolated_data_set = engineer.generate_features(data_set=isolated_data_set)
    processed_interleaved_data_set = engineer.generate_features(data_set=interleaved_data_set)

    # Selecting features based on correlation
    # todo work on the correlation feature selection
    list_of_features = selector.simple_correlation_selector(file_path=processed_isolated_data_set, threshold=0.95)

    # Impute data on the interleaved data set
    # todo work on the imputing module
    processed_interleaved_data_set = imputer.impute(data_set=processed_interleaved_data_set, method='')

    # Cluster values into groups
    # todo work on the clustering module
    clusters = clustering.cluster(file_path=processed_interleaved_data_set, clustering_method='')

    # Assign case id
    # todo work on the assigner module
    results_data_set = case_id_assigner.assigne_case_id(data_set=processed_interleaved_data_set, clusters=clusters)

    # Evaluate the case id assignment
    # todo work on the evaluation
    evaluation.print_evaluation(data_set=results_data_set)
