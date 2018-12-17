from __future__ import division
import numpy as np
import copy

from set_of_lines import SetOfLines
from set_of_points import SetOfPoints
import csv
import matplotlib.pyplot as plt
import time
from coreset_streamer import CoresetStreamer
from parameters_config import ParameterConfig
from sklearn.preprocessing import normalize
# from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn import preprocessing


class ExperimentsKMeansForLines:
    """
    A class that includes all the main experiments of the weighted centers corset API
    """

    # static variables
    current_iteration = 0
    parameters_config = None

    ######################################################################

    @staticmethod
    def EM_estimetor_k_means_for_lines(L, k, is_ground_truth=True, EM_iterations=np.nan, print_iterations=False):
        """
        This method gets a set L of lines, number of centers k>0, and returns the k centers that minimizes the sum
        of squared distances to L.
        Args:
            L (SetOfLines) : a set of lines
            k (int) : number of centers
            max_iterations (int) : number of maximum iterations allowed before convergence
            type: "novel" for our EM estimator, and "sota" for the state of the art EM estimator

        Returns:
            np.ndarray: a set of k centers that minimizes the sum of squared distances to L
        """

        size = L.get_size()
        dim = L.dim
        assert size > 0, "L is empty"
        assert k > 0, "k <= 0"

        bounds = ExperimentsKMeansForLines.bounds
        cost_min = np.infty
        parameter_config = ExperimentsKMeansForLines.parameters_config
        if is_ground_truth:
            #P = L.get_4_approx_points_ex_search(k)
            #return P
            L_size = L.get_size()
            EM_iterations = int(L_size * parameter_config.ground_true_iterations_number_ractor)
        size_of_sample = int(ExperimentsKMeansForLines.parameters_config.multiplications_of_k * k)
        if size <= size_of_sample:
            S = L
        else:
            S = L.get_sample_of_lines(size_of_sample)
        P_4_approx_min = S.get_4_approx_points(k)
        south_west = [bounds[0],bounds[1]]
        north_east = [bounds[2],bounds[3]]
        indexes_to_remove = []
        for j in range(P_4_approx_min.get_size()):
            p = P_4_approx_min.points[j]
            X = p[0]
            Y = p[1]
            if not (X >= float(south_west[0]) and X <= float(north_east[0]) and Y >= float(south_west[1]) and Y <= float(north_east[1])):
                indexes_to_remove.append(j)

        P_4_approx_min.remove_points_in_indexes(indexes_to_remove)
        if P_4_approx_min.get_size() > k:
            with open('output.file','w') as f:
                f.write(str(P_4_approx_min.points))
                f.write("\n***********************\n")
            #P_4_approx_min.points = np.delete(P_4_approx_min.points, indexes_to_remove, axis = 0)
            cost_min = L.get_sum_of_distances_to_centers(P_4_approx_min)
        for i in range(EM_iterations):
            if print_iterations:
                if i % 30 == 0 and i != 0:
                    print("EM estimator iteration number: ", i, "cost_min by now: ", cost_min)
            if size <= size_of_sample:
                S = L
            else:
                S = L.get_sample_of_lines(size_of_sample)
            P_4_approx_curent = S.get_4_approx_points(k)
            indexes_to_remove_curent = []
            for w in range(P_4_approx_curent.get_size()):
                p_curent = P_4_approx_curent.points[w]
                X = p_curent[0]
                Y = p_curent[1]
                if not (X >= float(south_west[0]) and X <= float(north_east[0]) and Y >= float(south_west[1]) and Y <= float(north_east[1])):
                    indexes_to_remove_curent.append(w)
            P_4_approx_curent.remove_points_in_indexes(indexes_to_remove_curent)
            if P_4_approx_curent.get_size() < k:
                continue
            with open('output.file','a') as f:
                f.write(str(P_4_approx_curent.points))
                f.write("\n***********************\n")
            cost_current = L.get_sum_of_distances_to_centers(P_4_approx_curent)
            if cost_current < cost_min:
                P_4_approx_min = P_4_approx_curent
                cost_min = cost_current

        return P_4_approx_min

    ######################################################################

    @staticmethod
    def EM_estimetor_k_means_for_points(P, is_ground_truth=True, EM_iterations=np.nan, is_print=False):
        """
        The state of the art algorithm to finding a robust median without outliers.
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers

        Returns:
            np.ndarray: the robust median of P
        """
        parameters_config = ExperimentsKMeansForLines.parameters_config
        centers_number = parameters_config.centers_number
        if is_ground_truth:
            P_size = P.get_size()
            EM_iterations = int( parameters_config.ground_true_iterations_number_ractor)
        min_k_centers = np.nan
        min_cost = np.infty
        for i in range(EM_iterations):
            if is_print:
                print("EM_iteration number ", i, ", out of ", EM_iterations, ", min_cost ", min_cost)
            k_centers = P.get_sample_of_points(centers_number)
            current_cost = k_centers.get_sum_of_distances_to_set_of_points(P)
            if current_cost < min_cost:
                min_cost = current_cost
                min_k_centers = k_centers
        return min_k_centers

    ######################################################################

    @staticmethod
    def RANSAC_k_means_for_lines(L, k, sample_size, required_time):
        """
        TODO: complete
        :param L:
        :param k:
        :return:
        """
        RANSAC_EM_ITERATIONS = ExperimentsKMeansForLines.parameters_config.RANSAC_EM_ITERATIONS
        random_sample_min = L.get_sample_of_lines(sample_size)  # sample m lines from uniform distribution on L
        random_sample_min_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(random_sample_min, k,
                                                                                             is_ground_truth=False,
                                                                                             EM_iterations=RANSAC_EM_ITERATIONS)  # send the sample to the EM estimator and get the k points that minimizes the sum of squared distances from the lines in the sample
        random_sample_min_cost = L.get_sum_of_distances_to_centers(random_sample_min_centers)
        total_time = 0
        RANSAC_steps = int(np.sqrt(sample_size))
        for i in range(RANSAC_steps):
            starting_time = time.time()
            random_sample_current = L.get_sample_of_lines(sample_size)  # sample m lines from uniform distribution on L
            random_sample_current_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(random_sample_current, k, is_ground_truth=False,EM_iterations=RANSAC_EM_ITERATIONS)  # send the sample to the EM estimator and get the k points that minimizes the sum of squared distances from the lines in the sample
            current_random_sample_cost = L.get_sum_of_distances_to_centers(random_sample_current_centers)
            if random_sample_min_cost > current_random_sample_cost:
                random_sample_min_cost = current_random_sample_cost  # save the minimum cost
                random_sample_min = random_sample_current
            ending_time = time.time()
            total_time += ending_time - starting_time
            if total_time >= required_time:
                break
        return random_sample_min, total_time

    ######################################################################

    @staticmethod
    def RANSAC_missing_entries(L, X, D, sample_size, missing_entries_alg, cost_type, required_time=np.nan):
        """
        TODO: complete
        :param L:
        :param k:
        :return:
        """
        parameters_config = ExperimentsKMeansForLines.parameters_config
        size = len(X)
        all_indices = np.asarray(range(size))
        random_sample_X_filled = current_random_sample_cost = random_sample_X_min = np.nan
        # RANSAC_steps = (sample_size / 10)
        random_sample_min_cost = np.infty
        RANSAC_EM_ITERATIONS = 10  # int(sample_size / 2)
        total_time = steps = 0
        while True:
            steps += 1
            starting_time = time.time()
            sample_indices = np.random.choice(all_indices, sample_size, False).tolist()
            random_sample_X_current = X[sample_indices]
            if missing_entries_alg == 'KNN':
                random_sample_X_filled = KNN(k=parameters_config.KNN_k).complete(
                    random_sample_X_current)  # Use 3 nearest rows which have a feature to fill in each row's missing features
            if missing_entries_alg == 'FancyImpute':
                random_sample_X_filled = SoftImpute().complete(random_sample_X_current)
            random_sample_X_filled = SetOfPoints(random_sample_X_filled)
            random_sample_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_points(random_sample_X_filled,
                                                                                              is_ground_truth=False,
                                                                                              EM_iterations=RANSAC_EM_ITERATIONS)
            if cost_type == 'matrices_diff':
                complete_X_by_random_sample = L.get_projected_centers(random_sample_centers)
                current_random_sample_cost = ((complete_X_by_random_sample - D.points) ** 2).mean()
            elif cost_type == 'distance_to_lines':
                current_random_sample_cost = L.get_sum_of_distances_to_centers(random_sample_centers)
            elif cost_type == 'k_means':
                current_random_sample_cost = D.get_sum_of_distances_to_set_of_points(random_sample_centers)

            if random_sample_min_cost > current_random_sample_cost:
                random_sample_min_cost = current_random_sample_cost  # save the minimum cost
                random_sample_X_min = L.get_lines_at_indices(sample_indices)
            ending_time = time.time()
            total_time += ending_time - starting_time
            if total_time >= parameters_config.coreset_to_ransac_time_rate * required_time:
                break
            # elif steps == RANSAC_steps:
            #    break
        return random_sample_X_min, total_time

    ######################################################################

    @staticmethod
    def normalized_lines_representation(spans, displacements):
        """
        This method gets a set of n lines represented by an array of n spanning vectors and an array od n displacements
        vectors, and returns these spanning vectors normalized and change each displacements in each line to be the
        closest point on the line to the origin. It is required in order to calculate all the distances later.
        Args:
            spans (np.ndarray) : an array of spanning vectors
            displacements (np.ndarray) : an array of displacements vectors

        Returns:
            spans_normalized, displacements_closest_to_origin (np.ndarray, np,ndarray) : the spanning vectors and the
                displacements vectors normalized and moved as required.
        """

        assert len(spans) > 0, "assert no spanning vectors"
        assert len(displacements) > 0, "assert no displacements vectors"
        assert len(spans) == len(displacements), "number of spanning vectors and displacements vectors not equal"

        dim = np.shape(spans)[1]

        spans_norms = np.sqrt(np.sum(spans ** 2, axis=1))
        spans_norms_repeat = np.repeat(spans_norms, dim, axis=0).reshape(-1, dim)
        spans_normalized = spans / spans_norms_repeat
        # print("spans_normalized: \n", spans_normalized)

        displacements_mul_minus_1 = displacements * -1
        displacements_mul_minus_1_mul_spans_normalized = np.sum(np.multiply(displacements_mul_minus_1, spans), axis=1)
        displacements_mul_minus_1_mul_spans_normalized_repeat_each_col = np.repeat(
            displacements_mul_minus_1_mul_spans_normalized, dim, axis=0).reshape(-1, dim)
        disp_mul_spans_normalized = np.multiply(spans_normalized,
                                                displacements_mul_minus_1_mul_spans_normalized_repeat_each_col)
        displacements_closest_to_origin = disp_mul_spans_normalized + displacements

        # print("displacements_closest_to_origin: \n", displacements_closest_to_origin)

        return spans_normalized, displacements_closest_to_origin

    ######################################################################

    @staticmethod
    def get_synthetic_data(lines_number):
        """
        Args:
            lines_number (int) : number of lines to generate

        Returns:
            SetOfLines: syntheticly generated set of lines_number lines
        """
        dim = 3

        outliers_lines_number = 4
        main_lines_number = lines_number - outliers_lines_number

        j = int(-1 * main_lines_number / 2)
        displacements = []
        for i in range(main_lines_number):
            if i == 0:
                displacements = np.array([[j, 0, 0]])
            else:
                displacements = np.concatenate((displacements, np.array([[j, 0, 0]])), axis=0)
            j = j + 1
        outliers_indexes = np.asarray(range(outliers_lines_number)) + 1
        for i in outliers_indexes:
            displacements = np.concatenate((displacements, np.array([[10000000 * i, 0, 0]])), axis=0)
        # displacements = np.concatenate((displacements, np.array([[10000000, 0, 0]])), axis=0)  # .reshape(-1,dim)
        # displacements = np.concatenate((displacements, np.array([[20000000, 0, 0]])), axis=0)  # .reshape(-1,dim)
        # displacements = np.concatenate((displacements, np.array([[30000000, 0, 0]])), axis=0)  # .reshape(-1,dim)
        # displacements = np.concatenate((displacements, np.array([[40000000, 0, 0]])), axis=0)  # .reshape(-1,dim)
        spans_befor_repeat = np.array([[0, 1, 0], [0, 0, 1]])
        spans = np.repeat(spans_befor_repeat.reshape(1, -1), int(lines_number / 2), axis=0).reshape(-1, dim)
        weights = np.ones(lines_number).reshape(-1, 1)
        spans_normalized, displacements_closest_to_otigin = ExperimentsKMeansForLines.normalized_lines_representation(
            spans, displacements)

        L = SetOfLines(spans_normalized, displacements_closest_to_otigin, weights)
        # L.plot_lines()
        return L

    ######################################################################

    @staticmethod
    def get_one_missing_entry_data(lines_number):
        """
        In this experiment, we get data D consist of real data observation from a file, remove randomly single entry
        from each observation and get an incomplete matrix X. We build from this matrix a sel L of n lines (each line
        corresponds to a row in X, and in parallel to the axes that correspond to the missing entry).
        :param lines_number:
        :return: L SetOfLines, X the incomplete matrix, D the complete matrix
        """
        indices_of_missing_entries = []
        all_points_for_corset = []
        X = []
        D = []
        parameters_config = ExperimentsKMeansForLines.parameters_config
        dim = parameters_config.dim
        header_indexes = parameters_config.header_indexes
        with open(parameters_config.input_points_file_name, 'rt') as csvfile:
            spamreader = list(csv.reader(csvfile))
            i = 0
            for row in spamreader:
                if row == []:
                    continue
                clean_row = [element.strip() for element in row]
                row_arr = np.asarray(clean_row)
                try:
                    row_arr_first_dim_cols = row_arr[header_indexes]
                except:
                    x = 2
                D.append(copy.deepcopy(row_arr_first_dim_cols))
                # row_arr_first_dim_cols[random_index] = '?'
                # indices_of_missing_entries.append(random_index)
                # all_points_for_corset.append(copy.deepcopy(row_arr_first_dim_cols))
                i += 1
                if i == int(lines_number):
                    break
            D = np.asarray([[float(i) for i in point] for point in D])
            X = copy.deepcopy(D)
            displacements = copy.deepcopy(D)
            indices_of_missing_entries = np.asarray(indices_of_missing_entries)
            spanning_vectors = np.zeros(shape=(lines_number, dim))  # in the beggining everything is full with zeros
            for i in range(lines_number):
                random_index = np.random.randint(0, dim)
                spanning_vectors[i][random_index] = 1
                try:
                    displacements[i][random_index] = '0'
                except:
                    x = 2
                X[i][random_index] = np.nan
            displacements = np.asarray([[float(i) for i in point] for point in displacements])
            # spans_normalized, displacements_closest_to_otigin = ExperimentsKMeansForLines.normalized_lines_representation(spanning_vectors, displacements)
            weights = np.ones(lines_number).reshape(-1, 1)
            L = SetOfLines(spanning_vectors, displacements, weights)
            # X = ExperimentsKMeansForLines.get_incomplete_matrix_from_lines(L)
            # D = np.asarray([[float(i) for i in point] for point in D])
            return L, X, D

    ######################################################################

    @staticmethod
    def get_incomplete_matrix_from_lines(L):
        """
        This method gets a set L of lines that each is in parallel to one of the axes, and returns an uncompleted
        matrix with the np.nan entry (missing value) in the feature axes the line is in parallel to.
        :param L:
        :return:
        """

        displacements = copy.deepcopy(L.displacements)
        spans = L.spans

        for i in range(len(displacements)):
            missing_entry_index = np.where(spans[i] == 1)
            displacements[i][missing_entry_index] = np.nan
        X = displacements
        return X

    ######################################################################

    @staticmethod
    def run_corset_and_RANSAC_streaming_synthetic(L, sample_size):
        """
        TODO: complete
        In this experiment, we get a set P consist of weighted points with outliers, sample randomly a set S_1
        of m points from P, and sample a coreset S_2 consist of m points from P. Run the the recursive median on S_1 to
        find a median q_1, and run the recursive median on S_2 to find a median q_2, and compare the robust cost from q_1
        to P (ignoring the k farthest points) versus the robust cost from q_2 to P (ignoring the k farthest points).
        Args:
            k (int) : number of weighted centers
            sample_size (int) : size of sample
        Returns:
            [float, float, float] : the coreset total cost, the RANSAC total cost, and the ground truth total cost
        """

        parameters_config = ExperimentsKMeansForLines.parameters_config
        random_sample_means = coreset_means = []
        lines_number = parameters_config.lines_number
        k = parameters_config.centers_number
        min_coreset_cost = np.infty
        coreset_total_time = 0
        for i in range(parameters_config.coreset_iterations):
            # print("started coreset")
            coreset_starting_time = time.time()
            C = CoresetStreamer(sample_size=sample_size, lines_number=lines_number, k=k,
                                parameters_config=parameters_config).stream(L)
            coreset_ending_time = time.time()
            coreset_total_time += coreset_ending_time - coreset_starting_time
            # print("finished coreset")
            # print("started to find means for coreset")
            coreset_means = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(C, k)
            # print("finished to find means for coreset")
            C.set_all_weights_to_specific_value(1.0)
            # print("coreset_means.points: \n", coreset_means.points)
            current_corset_cost = L.get_sum_of_distances_to_centers(coreset_means)
            if min_coreset_cost > current_corset_cost:
                min_coreset_cost = current_corset_cost
        coreset_cost = min_coreset_cost
        # print("coreset_cost: ", coreset_cost)

        RANSAC_total_time = np.nan
        min_random_sample_cost = np.infty
        if parameters_config.coreset_iterations == 0:
            coreset_total_time = 30
        for i in range(parameters_config.RANSAC_iterations):
            # print("started RANSAC")
            R, RANSAC_total_time = ExperimentsKMeansForLines.RANSAC_k_means_for_lines(L, k, sample_size,
                                                                                      coreset_total_time)
            # print("finished RANSAC")
            R.weights = R.weights.reshape(-1)
            # print("started to find means for RANSAC")
            random_sample_means = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(R, k)
            # print("finished to find means for RANSAC")
            # print("random_sample_means.points: \n", random_sample_means.points)
            current_random_sample_cost = L.get_sum_of_distances_to_centers(random_sample_means)
            if min_random_sample_cost > current_random_sample_cost:
                min_random_sample_cost = current_random_sample_cost
        random_sample_cost = min_random_sample_cost
        # print("random_sample_cost: ", random_sample_cost)

        return [coreset_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, coreset_means, random_sample_means]

    ######################################################################

    @staticmethod
    def get_coreset_and_ransac_means(L, experiment_type = 'google_maps'):
        ExperimentsKMeansForLines.parameters_config = ExperimentsKMeansForLines.init_parameter_config(experiment_type)
        parameters_config = ExperimentsKMeansForLines.parameters_config
        lines_number = parameters_config.lines_number
        # L = ExperimentsKMeansForLines.get_lines_from_file()
        #L.shuffle_lines()
        sample_size = int(L.get_size() / 10)
        if sample_size == 0:
            sample_size = L.get_size()
        [C_cost, random_sample_cost, coreset_total_time,
         RANSAC_total_time, coreset_means,
         random_sample_means] = ExperimentsKMeansForLines.run_corset_and_RANSAC_streaming_synthetic(L=L,sample_size=sample_size)
        # print("C_cost: ", C_cost)
        # print("random_sample_cost: ", random_sample_cost)
        return coreset_means, random_sample_means

    ######################################################################

    @staticmethod
    def get_coreset_means(L, experiment_type = 'google_maps'):
        ExperimentsKMeansForLines.parameters_config = ExperimentsKMeansForLines.init_parameter_config(experiment_type)
        ExperimentsKMeansForLines.parameters_config.RANSAC_iterations = 0
        # L = ExperimentsKMeansForLines.get_lines_from_file()
        #L.shuffle_lines()
        sample_size = int(L.get_size() / 10)
        if sample_size < 3:
            sample_size = L.get_size()        
        [C_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, coreset_means, random_sample_means] = ExperimentsKMeansForLines.run_corset_and_RANSAC_streaming_synthetic(L=L,sample_size=sample_size)
        # print("C_cost: ", C_cost)
        # print(coreset_means.points)
        return coreset_means
    ######################################################################

    @staticmethod
    def get_ransac_means(L, experiment_type = 'google_maps'):
        ExperimentsKMeansForLines.parameters_config = ExperimentsKMeansForLines.init_parameter_config(experiment_type)
        ExperimentsKMeansForLines.parameters_config.coreset_iterations = 0
        # L = ExperimentsKMeansForLines.get_lines_from_file()
        #L.shuffle_lines()
        sample_size = 7
        if sample_size > L.get_size():
            sample_size = L.get_size()        
        [C_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, coreset_means, random_sample_means] = ExperimentsKMeansForLines.run_corset_and_RANSAC_streaming_synthetic(L=L,sample_size=sample_size)
        # print("random_sample_cost: ", random_sample_cost)
        return random_sample_means

    ######################################################################

    @staticmethod
    def get_lines_from_file():

        parameters_config = ExperimentsKMeansForLines.parameters_config
        lines_nuber = parameters_config.lines_number
        file_name = parameters_config.input_points_file_name
        lines = []
        with open(file_name, 'rt') as csvfile:
            spamreader = csv.reader(csvfile)
            i = 0
            for row in spamreader:
                row_splited = [element.split(' ') for element in row]
                row_final = [float(entry) for entry in row_splited[0]]
                lines.append(row_final)
                i += 1
                if i == lines_nuber:
                    break

        L = SetOfLines(lines=lines, is_points=True)
        return L

    ######################################################################

    @staticmethod
    def run_corset_and_RANSAC_streaming_missing_entries(L, X, D, sample_size, missing_entries_alg, cost_type):
        """
        TODO: complete
        In this experiment, we get a set P consist of weighted points with outliers, sample randomly a set S_1
        of m points from P, and sample a coreset S_2 consist of m points from P. Run the the recursive median on S_1 to
        find a median q_1, and run the recursive median on S_2 to find a median q_2, and compare the robust cost from q_1
        to P (ignoring the k farthest points) versus the robust cost from q_2 to P (ignoring the k farthest points).
        Args:
            k (int) : number of weighted centers
            sample_size (int) : size of sample
        Returns:
            [float, float, float] : the coreset total cost, the RANSAC total cost, and the ground truth total cost
        """

        parameters_config = ExperimentsKMeansForLines.parameters_config
        D = SetOfPoints(D)
        lines_number = parameters_config.lines_number
        k = parameters_config.centers_number
        min_coreset_cost = np.infty
        coreset_total_time = 0
        current_corset_cost = np.nan
        # print("started coreset")
        for i in range(parameters_config.coreset_iterations):
            coreset_starting_time = time.time()
            C = CoresetStreamer(sample_size=sample_size, lines_number=lines_number, k=k,
                                parameters_config=parameters_config).stream(L)
            coreset_ending_time = time.time()
            coreset_total_time += coreset_ending_time - coreset_starting_time
            coreset_means = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(C, k)
            if cost_type == 'matrices_diff':
                complete_X_by_corset = L.get_projected_centers(coreset_means)
                current_corset_cost = ((complete_X_by_corset - D.points) ** 2).mean()
            elif cost_type == 'distance_to_lines':
                current_corset_cost = L.get_sum_of_distances_to_centers(coreset_means)
            elif cost_type == 'k_means':
                C_filled = C.get_projected_centers(coreset_means)
                C_filled = SetOfPoints(C_filled)  # SetOfPoints(C_filled, C.weights)
                C_filled_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_points(C_filled)
                # C_filled_centers.set_all_weights(1)
                current_corset_cost = D.get_sum_of_distances_to_set_of_points(C_filled_centers)
            if min_coreset_cost > current_corset_cost:
                min_coreset_cost = current_corset_cost
        coreset_cost = min_coreset_cost
        # print("finished coreset")
        # print("coreset_cost: ", coreset_cost)

        min_random_sample_cost = np.infty
        current_random_sample_cost = RANSAC_total_time = np.nan

        # print("started RANSAC")
        for i in range(parameters_config.RANSAC_iterations):
            random_sample, RANSAC_total_time = ExperimentsKMeansForLines.RANSAC_missing_entries(L, X, D, sample_size,
                                                                                                missing_entries_alg,
                                                                                                cost_type,
                                                                                                coreset_total_time)
            random_sample_means = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(random_sample, k)
            if cost_type == 'matrices_diff':
                complete_X_by_random_sample = L.get_projected_centers(random_sample_means)
                current_random_sample_cost = ((complete_X_by_random_sample - D.points) ** 2).mean()
            elif cost_type == 'distance_to_lines':
                current_random_sample_cost = L.get_sum_of_distances_to_centers(random_sample_means)
            elif cost_type == 'k_means':
                random_sample_filled = random_sample.get_projected_centers(random_sample_means)
                random_sample_filled = SetOfPoints(random_sample_filled)
                random_sample_filled_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_points(
                    random_sample_filled)
                current_random_sample_cost = D.get_sum_of_distances_to_set_of_points(random_sample_filled_centers)
            if min_random_sample_cost > current_random_sample_cost:
                min_random_sample_cost = current_random_sample_cost
        random_sample_cost = min_random_sample_cost
        # print("finished RANSAC")
        # print("random_sample_cost: ", random_sample_cost)

        return [coreset_cost, random_sample_cost, coreset_total_time, RANSAC_total_time]

    ######################################################################

    @staticmethod
    def error_vs_coreset_size_streaming(experiment_type='google_maps'):
        """
        In this experiment, we get an uncompleted matrix D with one randomly missing entry in each observation in D.
        We are running state of the art matrix completion algorithms on our coreset sample, and on a sample we get from
        RANSAC as well, and compare the solution accuracy versus the size of sample.
        :return:
        """
        ExperimentsKMeansForLines.parameters_config = ExperimentsKMeansForLines.init_parameter_config(experiment_type)
        parameters_config = ExperimentsKMeansForLines.parameters_config
        # print("*** cost_type: ", parameters_config.cost_type, ", missing_entries_alg: ",
        #       parameters_config.missing_entries_alg, ", centers_number: ", parameters_config.centers_number, " ***")
        # parameters
        lines_number = parameters_config.lines_number
        inner_iterations = parameters_config.inner_iterations

        # containers for statistics
        C_error_totals_final = []  # coreset total average error at each iteration
        C_error_totals_var = []  # coreset total variance for each sample
        random_sample_error_totals_final = []  # RANSAC total average error at each iteration
        random_sample_error_totals_var = []  # RANSAC total variance for each sample
        coreset_totals_time = []  # RANSAC total variance for each sample
        RANSAC_totals_time = []  # RANSAC total variance for each sample
        sample_sizes = []
        lines_numbers = []
        ground_truth_cost = 1
        L = X = D = C_cost = random_sample_cost = coreset_total_time = RANSAC_total_time = np.nan
        if experiment_type == 'synthetic':
            L = ExperimentsKMeansForLines.get_synthetic_data(lines_number)
        if experiment_type == 'google_maps':
            L = ExperimentsKMeansForLines.get_lines_from_file()
        if experiment_type == 'google_maps':
            L = ExperimentsKMeansForLines.get_lines_from_file()

        for u in range(len(parameters_config.sample_sizes)):
            ExperimentsKMeansForLines.current_iteration = u
            sample_size = parameters_config.sample_sizes[u]
            # print("main iteration number ", u)
            C_error_total = []
            random_sample_error_total = []
            coreset_total_time_inner = []
            RANSAC_total_time_inner = []
            for t in range(inner_iterations):
                # print("inner iteration number ", t)
                if experiment_type == 'google_maps':
                    [C_cost, random_sample_cost, coreset_total_time,
                     RANSAC_total_time, coreset_means, random_sample_means] = ExperimentsKMeansForLines.run_corset_and_RANSAC_streaming_synthetic(L=L,
                                                                                                              sample_size=sample_size)
                if experiment_type == 'mising_entries':
                    [C_cost, random_sample_cost, coreset_total_time,
                     RANSAC_total_time] = ExperimentsKMeansForLines.run_corset_and_RANSAC_streaming_missing_entries(L=L,
                                                                                                                    X=X,
                                                                                                                    D=D,
                                                                                                                    sample_size=sample_size,
                                                                                                                    missing_entries_alg=parameters_config.missing_entries_alg,
                                                                                                                    cost_type=parameters_config.cost_type)
                C_error = C_cost / ground_truth_cost
                random_sample_error = random_sample_cost / ground_truth_cost
                C_error_total.append(C_error)
                random_sample_error_total.append(random_sample_error)
                coreset_total_time_inner.append(coreset_total_time)
                RANSAC_total_time_inner.append(RANSAC_total_time)
            # avgs
            C_error_total_avg = np.mean(C_error_total)
            C_error_total_var = np.var(C_error_total) / C_error_total_avg
            random_sample_error_total_avg = np.mean(random_sample_error_total)
            random_sample_error_total_var = np.var(random_sample_error_total) / random_sample_error_total_avg
            coreset_total_time_inner_avg = np.mean(coreset_total_time_inner)
            RANSAC_total_time_inner_avg = np.mean(RANSAC_total_time_inner)
            C_error_totals_final.append(C_error_total_avg)
            C_error_totals_var.append(C_error_total_var)
            random_sample_error_totals_final.append(random_sample_error_total_avg)
            random_sample_error_totals_var.append(random_sample_error_total_var)
            coreset_totals_time.append(coreset_total_time_inner_avg)
            RANSAC_totals_time.append(RANSAC_total_time_inner_avg)
            sample_sizes.append(sample_size)
            lines_numbers.append(lines_number)
            # information printing
            # print("lines_numbers = ", lines_numbers)
            # print("sample_sizes = ", sample_sizes)
            # print("C_error_totals_final = ", C_error_totals_final)
            # print("random_sample_error_totals_final = ", random_sample_error_totals_final)
            # print("C_error_totals_var =  ", C_error_totals_var)
            # print("random_sample_error_totals_var = ", random_sample_error_totals_var)
            # print("coreset_totals_time = ", coreset_totals_time)
            # print("RANSAC_totals_time = ", RANSAC_totals_time)
            # print("coreset_to_RANSAC_time_rate = ",
            #       (np.asarray(coreset_totals_time) / np.asarray(RANSAC_totals_time)).tolist())
            # print("*** cost_type: ", parameters_config.cost_type, ", missing_entries_alg: ",
            #       parameters_config.missing_entries_alg, ", centers_number: ", parameters_config.centers_number, " ***")
        fig = plt.figure(1)
        ax_errors = fig.add_subplot(111)
        linestyle = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 1, "elinewidth": 1, "capsize": 4}
        ax_errors.errorbar(sample_sizes, C_error_totals_final, color='b', fmt='o', **linestyle)
        ax_errors.errorbar(sample_sizes, random_sample_error_totals_final, color='r', fmt='o', **linestyle)
        plt.ylabel('error')
        plt.xlabel('#sample size')
        plt.show()

    ######################################################################

    @staticmethod
    def init_parameter_config(experiment_type):
        """
        TODO: complete
        :param experiment_type:
        :return:
        """

        parameter_config = ParameterConfig()

        """

        if experiment_type == 'synthetic':
            # main parameters
            parameter_config.dim = len(parameter_config.header_indexes)
            parameter_config.lines_number = 20000

            # experiment  parameters
            parameter_config.sample_sizes = [50, 100, 200, 400, 500, 700, 1000, 1500, 2000]
            parameter_config.inner_iterations = 20
            parameter_config.centers_number = 3
            parameter_config.ground_true_iterations_number_ractor = 10



            # EM k means for lines estimator parameters
            parameter_config.multiplications_of_k = 2

            # k means for lines coreset parameters
            parameter_config.inner_a_b_approx_iterations = 1
            parameter_config.sample_rate_for_a_b_approx = 0.5

            # weighted centers coreset parameters
            parameter_config.median_sample_size = 1
            parameter_config.closest_to_median_rate = 0.5
            parameter_config.number_of_remains_multiply_factor = 1
            parameter_config.max_sensitivity_multiply_factor = 2

            # iterations
            parameter_config.RANSAC_iterations = 1
            parameter_config.coreset_iterations = 1
            parameter_config.RANSAC_EM_ITERATIONS = 10

            # files
            parameter_config.input_points_file_name = 'datasets/SimpleHome_XCS7_1002_WHT_Security_Camera/bengin_traffic.csv'
        """

        if experiment_type == 'google_maps':
            # main parameters
            parameter_config.header_indexes = [0, 1, 2]
            parameter_config.dim = len(parameter_config.header_indexes)
            parameter_config.lines_number = -1

            # coreset parameters
            parameter_config.a_b_approx_minimum_number_of_lines = 20
            parameter_config.sample_size_for_a_b_approx = 10

            # experiment  parameters
            parameter_config.sample_sizes = [500, 1000, 2000, 3000, 4000, 5000, 6000]
            parameter_config.inner_iterations = 10
            parameter_config.centers_number = 5

            # missing entries parameters
            parameter_config.missing_entries_alg = 'KNN'  # 'FancyImpute' #
            parameter_config.cost_type = 'k_means'  # 'matrices_diff' #
            parameter_config.KNN_k = 10

            # EM k means for lines estimator parameters
            parameter_config.multiplications_of_k = parameter_config.centers_number
            parameter_config.ground_true_iterations_number_ractor = 50


            # weighted centers coreset parameters
            parameter_config.median_sample_size = 1
            parameter_config.closest_to_median_rate = 0.3
            parameter_config.number_of_remains_multiply_factor = 1
            parameter_config.number_of_remains = 10
            parameter_config.max_sensitivity_multiply_factor = 2

            # iterations
            parameter_config.RANSAC_iterations = 1
            parameter_config.coreset_iterations = 1
            parameter_config.RANSAC_EM_ITERATIONS = 1
            parameter_config.coreset_to_ransac_time_rate = 1

            # files
            parameter_config.input_points_file_name = 'china-latest111111_shuffled.csv'

        return parameter_config



