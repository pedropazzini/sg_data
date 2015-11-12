import abc
import sys

import numpy as np

from classes.enums import Normalization

class Clustering:

    def __init__(self, data, expected, k_vector, algorithm_vector, distance_vector, normalize_type):

        self.k_vector = k_vector
        self.algorithm_vector = algorithm_vector
        self.distance_vector = distance_vector
        self.data = data

        self.normalize_type = normalize_type
        self.is_normalized = False

        self.n_expected, self.expected_result = self.transform_expected_result_to_solution(expected)

        self.expected_solutions = expected

    def transform_expected_result_to_solution(self, expected):

        s = {}
        n = 0
        for key,val in expected.items():
            v = int(val)
            if v  not in s:
                s[v] = [key]
                n += 1
            else:
                s[v].append(key)
        return n,s


    @abc.abstractmethod
    def fit(self,print_iterations = False):
        """ Method to fit the data of the clustering algorithm with the parameters passed in the constructor.
        If is necessary the printing of some informations during the fitting the parameter 'print_iterations' should 
        be true. """
        print("Abstract")
        return


    def normalize(self):
        """ Methos that normalizes each curve of the data."""

        if (self.normalize_type is Normalization.none):
            return # Normalization should not be done

        new_data = {}
        for key, vals in self.data.items():
            if (self.normalize_type is Normalization.by_Z_normalization):
                mean_ = np.mean(vals)
                std_ = np.std(vals)
                new_data[key] = (vals - mean_)/std_
            else:
                max_val = sys.float_info.min
                min_val = sys.float_info.max
                for val in vals:
                    if (val > max_val):
                        max_val = val
                    if (val < min_val):
                        min_val = val
                if (self.normalize_type is Normalization.by_max_min):
                    new_data[key] = (vals-max_val)/(min_val-max_val) 
                elif (self.normalize_type is Normalization.by_max):
                    new_data[key] = vals/max_val
                else:
                    raise ValueError('Value not accepted.')

        self.data = new_data

        self.is_normalized = True

    def get_matrix_similarity(similarity_name):

        if (not self.is_normalized):
            self.normalize()


    #def fit_all(self, print_iterations = False):
     #   for k in k_vector:
      #      for alg in algorithm:
       #         for dist in distance_vector:

