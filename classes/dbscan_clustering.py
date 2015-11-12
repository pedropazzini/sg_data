import numpy as np
from sklearn.cluster import DBSCAN
from classes.clustering import Clustering

class DBSCAN_clustering(Clustering):

    def __init__(self, data, expected_result, k_vector, algorithm_vector, distance_vector, normalization_type, max_iter=30, keep_solutions = False):

        super(K_means_clustering,self).__init__(data,expected_result,k_vector,algorithm_vector, distance_vector, normalization_type)

        self.labels_predicted = np.array([])

        self.algorithm_name = 'DBSCAN'

    def fit(self, distance, normalization_type):


