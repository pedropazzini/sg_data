import numpy as np
from classes.k_means_clustering import K_means_clustering

r = open('dados.txt','r').read()
dic = eval(r)

sazonal_data = {}
total_data = 0
max_data = 64*3
for meter, data in dic.items():
    if (data is None):
        continue
    if (total_data >= max_data):
        break
    #sazonal_data[meter] = {} 
    for season, vals in data.items():
        if (season is not 'S'):
            continue
        l = []
        sorted_dict = {}
        hour = 0
        minute = 0
        while (hour < 24):
            time = str(hour) + ":" + str(minute)

            measure = vals[time]
            l.append(measure)

            hour = hour if minute == 0 else hour + 1
            minute = 0 if minute == 30 else 30

        sazonal_data[meter] = np.array(l)
        total_data += 1
        if (total_data >= max_data):
            break

k = K_means_clustering(sazonal_data,list(range(2,18)),['k-means','k-means++'],['euclidean','cityblock','chebyshev','DTW', 'LB_Keogh'],normalize_by_min=False)
#k = K_means_clustering(sazonal_data,list(range(2,3)),['k-means'],['euclidean'],normalize_by_min=False)
k.fit(verbose=True)
k.plot_all(as_time_series=True,key_name=(str(max_data) + "_sg_"))
k.plot_silhouette(key_name=(str(max_data) + "_sg_"))
