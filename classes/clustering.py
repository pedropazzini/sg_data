import abc
import sys

class Clustering:

    def __init__(self, data, k_vector, algorithm_vector, distance_vector, normalize_by_min, normalize_by_max):

        self.k_vector = k_vector
        self.algorithm_vector = algorithm_vector
        self.distance_vector = distance_vector
        self.data = data

        self.normalize_by_max = normalize_by_max
        self.normalize_by_min = normalize_by_min

    @abc.abstractmethod
    def fit(self,print_iterations = False):
        """ Method to fit the data of the clustering algorithm with the parameters passed in the constructor.
        If is necessary the printing of some informations during the fitting the parameter 'print_iterations' should 
        be true. """
        print("Abstract")
        return


    def normalize(self):
        """ Methos that normalizes each curve of the data."""
        if (self.normalize_by_min is None or self.normalize_by_max is None or (not self.normalize_by_max)):
            raise ValueError("'normalize_by_min' and 'normalize_by_max' not set on constructor")

        new_data = {}
        for key, vals in self.data.items():
            max_val = sys.float_info.min
            min_val = sys.float_info.max
            for val in vals:
                if (val > max_val):
                    max_val = val
                if (val < min_val):
                    min_val = val
            if (self.normalize_by_min):
                new_data[key] = (vals-max_val)/(min_val-max_val) 
            else:
                new_data[key] = vals/max_val

        self.data = new_data


    #def fit_all(self, print_iterations = False):
     #   for k in k_vector:
      #      for alg in algorithm:
       #         for dist in distance_vector:

