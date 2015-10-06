import sys
from classes.k_means_clustering import K_means_clustering
import numpy as np

benchmark = sys.argv[1]
folder_name = 'time_series_benchmark/UCR_TS_Archive_2015/'

i = 0
data = {}
expected = {}

clusters_labels = []

with open(folder_name + benchmark + '/' + benchmark + '_TRAIN', 'r') as f:
    for line in f:
        i += 1
        floats = [float(x) for x in line.split(',')]
        print(floats)
        data[i] = np.array(floats[1:len(floats)])
        label = floats[0]
        expected[i] = label
        if label not in clusters_labels:
            clusters_labels.append(label)

n_expected_cluster = len(clusters_labels)
diff = 4
min_k = 2 if n_expected_cluster - diff < 2 else n_expected_cluster - diff
max_k = n_expected_cluster + diff
algs = ['k-means','k-means++']
#algs = ['k-means++']
#dists = ['euclidean','cityblock','chebyshev','DTW','LB_Keogh']
dists = ['euclidean','cityblock','chebyshev']
#ks = list(range(2,15))
ks = list(range(min_k,max_k))
normal_by_min = False
k = K_means_clustering(data,ks,algs,dists,normalize_by_min = normal_by_min,keep_solutions = True )
k.fit(verbose=True)
k.plot_all(as_time_series=True,key_name=benchmark)
k.plot_validations(key_name=benchmark)
k.plot_real_solution(len(clusters_labels),expected,'k-means','euclidean',benchmark)
k.plot_real_solution(len(clusters_labels),expected,'k-means++','euclidean',benchmark)
