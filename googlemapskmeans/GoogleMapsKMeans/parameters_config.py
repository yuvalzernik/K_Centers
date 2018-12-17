from __future__ import division


class ParameterConfig:
    def __init__(self):
        # main parameters
        self.header_indexes = [34]
        self.dim = len(self.header_indexes)
        self.lines_number = 40000

        # experiment  parameters
        self.sample_sizes = [100, 200, 500, 700, 1000]
        self.inner_iterations = 5
        self.centers_number = 15
        self.outliers_trashold_value = 300000

        #coreset parameters
        self.a_b_approx_minimum_number_of_lines = 2000
        self.sample_size_for_a_b_approx = 500

        # EM k means for lines estimator parameters
        self.multiplications_of_k = 2
        self.EM_iteration_test_multiplications = 20

        # EM k means for points estimator parameters
        self.ground_true_iterations_number_ractor = 50

        # weighted centers coreset parameters
        self.median_sample_size = 1
        self.closest_to_median_rate = 0.5
        self.number_of_remains_multiply_factor = 1
        self.number_of_remains = 100
        self.max_sensitivity_multiply_factor = 2

        # iterations
        self.RANSAC_iterations = 1
        self.coreset_iterations = 1
        self.RANSAC_EM_ITERATIONS = 10
        self.coreset_to_ransac_time_rate = 0.1

        # files
        self.input_points_file_name = 'datasets/SimpleHome_XCS7_1002_WHT_Security_Camera/bengin_traffic_normalized.csv'

        #missing entries parameters
        self.missing_entries_alg = 'KNN'
        self.cost_type= 'matrices_diff'
        self.KNN_k = 3

        #data handler parameters
        self.points_number = 10
        self.output_file_name = 'datasets/bengin_traffic_normalized.csv'

