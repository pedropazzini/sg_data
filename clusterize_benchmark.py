import sys
from classes.k_means_clustering import K_means_clustering
import numpy as np

benchmark = sys.argv[1]
folder_name = 'time_series_benchmark/UCR_TS_Archive_2015/'

print (benchmark)
with open(folder_name + benchmark + '/' + benchmark + '_TRAIN', 'r') as f:
    i = 0
    data = {}
    expected = {}
    for line in f:
        i += 1
        floats = [float(x) for x in line.split(',')]
        print(floats)
        data[i] = np.array(floats[1:len(floats)])
        expected[i] = floats[0]

k = K_means_clustering(data,list(range(2,5)),['k-means','k-means++'],['euclidean','cityblock','chebyshev','DTW','LB_Keogh'])
#k = K_means_clustering(data,list(range(2,10)),['k-means'],['euclidean'])
k.fit(verbose=True)
k.plot_all(as_time_series=True,key_name=benchmark)
k.plot_silhouette(key_name=benchmark)
