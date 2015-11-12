import sys
from classes.k_means_clustering import K_means_clustering
import numpy as np
from classes.enums import Normalization

benchmark_list_str = sys.argv[1]
benchmark_list = benchmark_list_str.split(',')
for benchmark in benchmark_list:

    results = {}
    max_it = 10

    for it in range(max_it):

        folder_name = 'time_series_benchmark/UCR_TS_Archive_2015/'

        i = 0
        data = {}
        expected = {}

        clusters_labels = []

        with open(folder_name + benchmark + '/' + benchmark + '_TRAIN', 'r') as f:
            for line in f:
                i += 1
                floats = [float(x) for x in line.split(',')]
                data[i] = np.array(floats[1:len(floats)])
                label = floats[0]
                expected[i] = label
                if label not in clusters_labels:
                    clusters_labels.append(label)

        n_expected_cluster = len(clusters_labels)
        diff = 1
        min_k = 2 if n_expected_cluster - diff < 2 else n_expected_cluster - diff
        max_k = n_expected_cluster + diff
        algs = ['k-means']
        #dists = ['minkowski_1','minkowski_2','chebyshev','DTW','LB_Keogh','correlation_1_1.5','cort-euclidean']
        dists = ['LB_Keogh','correlation_1_1.5','cort-euclidean','DTW']
        ks = list(range(min_k,max_k))
        normal_by_min = False
        k = K_means_clustering(data,expected,[n_expected_cluster],algs,dists,Normalization.by_Z_normalization )
        k.fit(verbose=True)
        #k.plot_all(as_time_series=True,key_name=benchmark)
        #k.plot_validations(benchmark,n_expected_cluster)
        #k.plot_real_solution(len(clusters_labels),expected,'k-means','minkowski_1',benchmark)
        #k.plot_real_solution(len(clusters_labels),expected,'k-means++','minkowski_2',benchmark)

        if (benchmark not in results):
            results[benchmark] = {}
        results[benchmark][it] = (k.solutions, k.time_clustering, k.expected_indices)
