from classes.clustering import Clustering
from classes.dtw import DTW
from scipy.spatial import distance
from copy import copy

import numpy as np
import pandas as pd
import colorsys
import sys
import pdb
import time

import matplotlib.pyplot as plt

class K_means_clustering(Clustering):

    def __init__(self, data, k_vector, algorithm_vector, distance_vector,max_iter=10, keep_solutions = False, normalize_by_min=True, normalize_by_max=True):

        super(K_means_clustering,self).__init__(data,k_vector,algorithm_vector, distance_vector, normalize_by_min, normalize_by_max)

        self.max_iter = max_iter
        self.keep_solutions = keep_solutions

        self.iter_solutions = {}

        self.best_solution = None
        self.best_fitness = sys.float_info.max

        self.solutions = {}
        self.validations = {}

        self.time_clustering ={}
        self.time_validating = {}

        self.matrix_data = None
        self.mins = None
        self.maxs = None

        self.data_frame = pd.DataFrame(self.data)

    def gen_matrix_data(self):
        if not self.matrix_data:
            self.matrix_data = np.array([k for k in self.data.values()])
            self.mins = np.min(self.matrix_data)
            self.maxs = np.max(self.matrix_data)

    def fit(self,verbose=False):
        self.normalize()
        for k in self.k_vector:
            for dist in self.distance_vector:
                for algorithm in self.algorithm_vector:
                    start_time = time.time() # start measuring time
                    if(verbose):
                        info = "algorithm=%s, k=%s, distance=%s" % (algorithm,k, dist)#
                        print(info)
                    if (algorithm == 'k-means'):
                        self.fit_by_k__means(k,dist,verbose)

                    elif (algorithm == 'k-means++'):
                        self.fit_by_k_means_pp(k,dist,verbose)

                    self.add_time_clustering(algorithm,dist,k,start_time)

                    self.add_solution(algorithm,dist,k)

                    self.validate_solution(algorithm,dist,k, verbose)

    def add_time_clustering(self, algorithm, dist, k, start_time):

        end_time = time.time()

        if (algorithm not in self.time_clustering):
            self.time_clustering[algorithm] = {}

        if (dist not in self.time_clustering[algorithm]):
            self.time_clustering[algorithm][dist] = {}

        self.time_clustering[algorithm][dist][k] = end_time -start_time


    def add_solution(self,algorithm,dist,k):

        if (algorithm not in self.solutions):
            self.solutions[algorithm] = {}

        if(dist not in self.solutions[algorithm]):
            self.solutions[algorithm][dist] = {}

        self.solutions[algorithm][dist][k] = (self.best_fitness,self.best_solution)

    def validate_solution(self,algorithm,dist,k, verbose):

        #TODO: Implement other validation methods
        self.validate_solution_by_silhouette(algorithm,dist,k, verbose)

        self.validate_solution_by_CDI(algorithm,dist,k,verbose)

        self.validate_solution_by_MIA(algorithm,dist,k,verbose)

    def validate_solution_by_MIA(self,algorithm,dist,k,verbose):

        mia_val = float('inf')
        cluster_data = self.solutions[algorithm][dist][k]
        fitness = cluster_data[0]
        if (verbose):
            print("Calculating MIA of:")
            print("Algorithm="+algorithm+", Distance=" + dist + ",K=" + str(k))
            print ("Fitness = " + str(fitness))
        cluster = cluster_data[1]
        #method = getattr(distance,dist)
        K = 0
        d_hat = 0
        representative_loads = {}

        # Gets the distance between load diagrams in the same cluster
        for c_id, points in cluster.items():
            if (len(points) > 0):
                K += 1
            else:
                continue
            cluster_size = 0
            d = 0
            l = []
            for pointA in points:
                l.append(self.data[pointA])

            representative_loads[c_id] = np.mean(l,axis=0)

        d_hat = 0
        for c_id, points in cluster.items():
            n = 0
            d = 0
            for point in points:
                n += 1
                #TODO: Check if this value is to squared
                # TODO: Chek if should not use the same metric distance of the algorithm and not chicco distance
                #d += self.get_chicco_distance_between_loads(representative_loads[c_id],self.data[point])
                d += self.get_distance(dist, representative_loads[c_id],self.data[point])**2
            #TODO: Check if this value is to squared
            d_hat += np.sqrt((1/n)*d)**2

        #TODO: Check if this value is to squared
        mia_val = np.sqrt((1/K)*d_hat)

        self.add_validation('MIA',algorithm,dist,k,mia_val)

    def validate_solution_by_CDI(self,algorithm,dist,k,verbose):

        cdi_val = float('inf')
        cluster_data = self.solutions[algorithm][dist][k]
        fitness = cluster_data[0]
        if (verbose):
            print("Calculating CDI of:")
            print("Algorithm="+algorithm+", Distance=" + dist + ",K=" + str(k))
            print ("Fitness = " + str(fitness))
        cluster = cluster_data[1]
        #method = getattr(distance,dist)
        K = 0
        d_hat = 0
        representative_loads = {}

        # Gets the distance between load diagrams in the same cluster
        for c_id, points in cluster.items():
            if (len(points) > 0):
                K += 1
            else:
                continue
            cluster_size = 0
            d = 0
            l = []
            for pointA in points:
                PA = self.data[pointA]
                l.append([PA])
                for pointB in points:
                    #TODO: Check if this value is to squared
                    # TODO: Chek if should not use the same metric distance of the algorithm and not chicco distance
                    #d += self.get_chicco_distance_between_loads(PA,self.data[pointB])
                    d += self.get_distance(dist,PA,self.data[pointB])**2
                    cluster_size += 1
            representative_loads[c_id] = np.mean(l,axis=0)
            #TODO: Check if this value is to squared
            d_hat += np.sqrt(d/(2*cluster_size))**2

        d = 0
        cluster_size = 0
        d_hat_repr = 0
        for c_id_1, representative_load_1 in representative_loads.items():
            for c_id_2, representative_load_2 in representative_loads.items():
                #TODO: Check if this value is to squared
                # TODO: Chek if should not use the same metric distance of the algorithm and not chicco distance
                #d += self.get_chicco_distance_between_loads(representative_load_1,representative_load_2)
                d += self.get_distance(dist,representative_load_1,representative_load_2)**2
                cluster_size += 1

        #TODO: Check if this value is to squared
        d_hat_repr = np.sqrt(d/(2*cluster_size))**2

        cdi_val = np.sqrt(d_hat/K)/d_hat_repr

        self.add_validation('CDI',algorithm,dist,k,cdi_val)

    def get_chicco_distance_between_loads(self,l1,l2):
        '''
        Distance between two elecric curve loads (time series as presented at:
        Customer Characterization Options for Improvingi the Tariff Offer
        Gianfranco Chicco, Roberto Napoli, Petru Postolache,
        Mircea Scutariu and Cornel Toader
        '''
        H_1 = len(l1)
        H_2 = len(l2)
        if (H_1 != H_2):
            raise ValueError("Loads should be of the same size")

        return np.sqrt(np.sum(np.power(l1-l2,2))/H_1)

    def validate_solution_by_silhouette(self, algorithm, dist, k, verbose):

        s_is = {}
        cluster_data = self.solutions[algorithm][dist][k]
        fitness = cluster_data[0]
        if (verbose):
            print("Calculating Silhouette of:")
            print("Algorithm="+algorithm+", Distance=" + dist + ",K=" + str(k))
            print ("Fitness = " + str(fitness))
        cluster = cluster_data[1]
        #method = getattr(distance,dist)
        for c_id, points in cluster.items():
            a_vec = []
            for pointA in points:
                for pointB in points:
                    if (pointA == pointB):
                        continue
                    #a_vec.append(method(self.data[pointA],self.data[pointB]))
                    a_vec.append(self.get_distance(dist,self.data[pointA],self.data[pointB]))
                a_i = np.mean(a_vec)
                b_i = sys.float_info.max
                for c_id2, points2 in cluster.items():
                    b_vec = []
                    if (c_id == c_id2):
                        continue
                    else:
                        for pointC in points2:
                            #b_vec.append(method(self.data[pointA],self.data[pointC]))
                            b_vec.append(self.get_distance(dist,self.data[pointA],self.data[pointC]))

                        di_C = np.mean(b_vec)
                        if (di_C < b_i):
                            b_i = di_C

                s_i = 0
                if (a_i < b_i):
                    s_i = 1-a_i/b_i
                elif (a_i > b_i):
                    s_i = b_i/a_i -1

                s_is[pointA] = s_i

        silhouette = np.mean(list(s_is.values()))

        self.add_validation('silhouette',algorithm,dist,k,silhouette)

    def add_validation(self,validation_alg, algorithm,dist,k,validation_val):

        if (validation_alg not in self.validations):
            self.validations[validation_alg] = {}

        if (algorithm not in self.validations[validation_alg]):
            self.validations[validation_alg][algorithm] = {}

        if(dist not in self.validations[validation_alg][algorithm]):
            self.validations[validation_alg][algorithm][dist] = {}

        self.validations[validation_alg][algorithm][dist][k] = validation_val

    def fit_by_k__means(self,k,dist, verbose, is_kmeans_pp=False):
        for i in range(self.max_iter):
            if (verbose):
                print("Iter: %s/%s"%(i,self.max_iter))
            solution = self.generate_k_means_pp_solution(dist,k) if is_kmeans_pp else self.generate_random_solution(k)
            fitness, solution = self.converge_solution(dist,verbose,solution)
            if (self.keep_solutions):
                self.iter_solutions[i] = (fitness,solution)

            if(fitness < self.best_fitness):
                self.best_fitness = fitness
                self.best_solution = copy(solution)


    def converge_solution(self,dist,verbose,solution):

        convergence = False
        iters = 0
        max_iters = 100
        while (not convergence):
            new_solution = {}
            if (-1 in solution):
                pdb.set_trace()
            centroids = self.get_centroids(dist,solution)

            fitness = 0
            for key, val in self.data.items():
                min_distance = sys.float_info.max
                best_centroid = -1

                #Searches for the closest centroid
                for centroid_num,centroid_pos in centroids.items():
                    if (val.shape == centroid_pos.shape):
                        #d = method(val,centroid_pos)
                        d = self.get_distance(dist,val,centroid_pos)
                        if (d < min_distance):
                            min_distance = d
                            best_centroid = centroid_num
                    else:
                        if (verbose):
                            print("TODO: Invalid shape key[" + str(key) + "], pos[" + str(centroid_pos)  + "]:" + str(val.shape) + "/" + str(centroid_pos.shape))
                            if (len(centroid_pos) ==1):
                                raise ValueError("Excecao")

                            print(centroids)

                # Computes the fitness and adds the centroid to the closest cluster
                fitness += min_distance if min_distance != sys.float_info.max else 0
                if (best_centroid in new_solution):
                    new_solution[best_centroid].append(key)
                elif (best_centroid != -1):
                    new_solution[best_centroid] = list([key])

            iters += 1
            if (verbose):
                print("fitness=" + str(fitness) + ", iter=" + str(iters))

            # Checks the stop criteria which happens when there is no change in the solution or the max iters is reached
            if (solution == new_solution) or (iters > max_iters):
                convergence = True
            else:
                solution = copy(new_solution)

        return fitness,solution

    def get_centroids(self,dist,solution):
        centroids = {}
        for key,val in solution.items():
            centroid = self.get_centroid(val,dist)
            centroids[key] = centroid
        return centroids

    def get_centroid (self, key_series, dist):
        vals = np.array([])
        first = True
        second = False
        for i in key_series:
            if (first):
                vals = np.append(vals,self.data[i])
                first = False
                second = True
            else:
                if (second):
                    vals = np.append([vals],[self.data[i]],axis = 0)
                    second = False
                else:
                    if (vals[0].shape == self.data[i].shape):
                        vals = np.append(vals,[self.data[i]],axis = 0)

        # TODO: Verificar se a média faz sentido para outras distâncias que não a Euclidiana
        return np.mean(vals,axis=0) if len(vals.shape) > 1 else vals

    def get_dists_from_centers(self, centers, dist_type):

        dist = pd.DataFrame()
        for i in range(len(centers)):
            a = np.array([self.get_distance(dist_type,centers[i],self.data_frame.iloc[:,j]) for j in range(self.data_frame.shape[1])])
            dist.insert(0,i,a)

        return dist.min(axis = 1)


    def generate_k_means_pp_solution(self, dist_type, k):
        total_len_data = len(self.data.keys())
        #self.gen_matrix_data()
        # Generates initial center
        c1 = np.random.uniform(np.min(self.data_frame,axis =1),np.max(self.data_frame,axis=1))
        centers = list([c1])

        # Generates the other centers
        for i in range(1,k):
            dists = self.get_dists_from_centers(centers,dist_type)
            normalized_distances = dists / dists.sum()
            normalized_distances.sort()
            dice_roll = np.random.rand()
            min_over_roll = normalized_distances[normalized_distances.cumsum() >= dice_roll].min()
            index = normalized_distances[normalized_distances == min_over_roll].index[0]
            new_center = np.array(self.data_frame.iloc[:,index])
            centers.append(new_center)

        return self.get_solution_of_centroids(centers,dist_type)

    def get_solution_of_centroids(self, centroids, dist_type):

        solution = {}
        for key,val in self.data_frame.iteritems():
            np_d_array = np.array(val)

            c = 0
            min_dist = sys.float_info.max
            k = -1
            for centroid in centroids:
                dist = self.get_distance(dist_type,centroid,np_d_array)
                if (dist < min_dist):
                    k = c
                    min_dist = dist

                c += 1

            if (k in solution):
                solution[k].append(key)
            else:
                solution[k] = list([key])

        print(solution)
        return solution

    def generate_random_solution(self,k):
        total_len_data = len(self.data.keys())
        solution = {}
        for key in self.data.keys():
            label = np.random.randint(k)
            if (label in solution):
                solution[label].append(key)
            else:
                solution[label] = list([key])

        return solution

    def get_distance(self, dist, s1, s2):
        if (dist == 'DTW' or dist == 'LB_Keogh'):
            dtw_obj = DTW(s1,s2)
            if (dist == 'LB_Keogh'):
                return dtw_obj.get(is_LB_keogh=True)        
            else:
                return dtw_obj.get()        
        else:
            method = getattr(distance,dist)
            return method(s1,s2)



    def fit_by_k_means_pp(self, k, dist, verbose):

        self.fit_by_k__means(k, dist, verbose, is_kmeans_pp = True)

    def plot_clusters(self, algorithm, distance, k, key_name):

        cluster = self.solutions[algorithm][distance][k]
        colors=np.random.rand(k,)
        fitness = cluster[0]
        fig = plt.figure()
        for c, pts in cluster[1].items():
            x = []
            y = []
            for pt in pts:

                pos = self.data[pt]
                if (len(pos) != 2):
                    raise ValueError("Only 2-D plots allawed")

                x.append(pos[0])
                y.append(pos[1])

            plt.plot(x,y,'o')
        silhouette = self.validations['silhouette'][algorithm][distance][k]
        title = algorithm + ", d=" + distance + ", k=" + str(k) + ", silh.=" + str(silhouette)
        plt.title(title)
        plt.show()
        plt.savefig("./plots/" + key_name + "_" + algorithm + "_" + distance + "_" + str(k) + ".png")
        plt.close(fig)

    def plot_clusters_as_time_series(self,algorithm,distance,k, key_name):

        cluster = self.solutions[algorithm][distance][k]
        colors=np.random.rand(k,)
        fitness = cluster[0]
        for c, pts in cluster[1].items():
            l = []
            N = 0
            f, axarr = plt.subplots(2, sharex=True)
            for pt in pts:
                t_series = self.data[pt]
                axarr[1].plot(t_series)
                l.append([t_series])
                N += 1
            mean = np.mean(l,axis=0)
            std = np.std(l,axis=0)
            min_ = np.min(l,axis=0)
            max_ = np.max(l,axis=0)

            axarr[0].plot(mean[0],label='mean',color='blue')
            axarr[0].plot(min_[0],color='red')
            axarr[0].plot(max_[0],label='min/max',color='red')
            axarr[0].plot(mean[0] + std[0],color='green')
            axarr[0].plot(mean[0] - std[0],label='std',color='green')

            axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
            silhouette = self.validations['silhouette'][algorithm][distance][k]
            title = algorithm + ", d=" + distance + ", (c/k)=(" + str(c) + "/" + str(k) + "), silh.=" + str(silhouette) + ", N_curves=" + str(N)
            axarr[1].set_title(title)
            plt.show()
            plt.savefig("./plots/" + key_name  + "_time_series__" + algorithm + "_" + distance +  "_" + str(k) + "_" + str(c) + ".png")
            plt.close(f)


    def plot_all(self, as_time_series = False,key_name='sg'):

        for alg, d0 in self.solutions.items():
            for dist, d1 in d0.items():
                for k, d2 in d1.items():
                    if (as_time_series):
                        self.plot_clusters_as_time_series(alg,dist,k,key_name)
                    else:
                        self.plot_clusters(alg,dist,k, key_name)

    def plot_validation(self, key_name, validation_alg):
        fig = plt.figure()
        for alg,d0 in self.solutions.items():
            for dist,d1 in d0.items():
                s = []
                ks = []
                for k,d2 in d1.items():
                    silhouette = self.validations[validation_alg][alg][dist][k]
                    s.append(silhouette)
                    ks.append(k)

                plt.plot(ks,s, label= alg + "," + dist)

        plt.legend(loc="best")
        title = validation_alg + " x K"
        plt.title(title)
        plt.show()

        plt.savefig("./plots/" + key_name + "_" + validation_alg  + "_curves.png")
        plt.close(fig)

    def plot_validations(self,key_name):

        validation_algorithms = ['silhouette','CDI','MIA']
        for va in validation_algorithms:
            self.plot_validation(key_name,va)

    def plot_real_solution(self, n_expected_clusters, expected_solutions, algorithm,dist, key_name):

        cluster = self.solutions[algorithm][dist][n_expected_clusters]
        if not cluster:
            raise ValueError("Cluster no found for parameters: {'Algorithm':%s,'Distance':%s,'N_expected_clustes':%s"%(algorithm,dist,n_expected_clusters))

        for i in range(1,n_expected_clusters+1):
            keys = [k for k,v in expected_solutions.items() if v == i]
            l = []
            f, axarr = plt.subplots(2, sharex=True)
            N = 0
            for key in keys:
                t_series = self.data[key]
                axarr[1].plot(t_series)
                l.append([t_series])
                N += 1
            mean = np.mean(l,axis=0)
            std = np.std(l,axis=0)
            min_ = np.min(l,axis=0)
            max_ = np.max(l,axis=0)

            axarr[0].plot(mean[0],label='mean',color='blue')
            axarr[0].plot(min_[0],color='red')
            axarr[0].plot(max_[0],label='min/max',color='red')
            axarr[0].plot(mean[0] + std[0],color='green')
            axarr[0].plot(mean[0] - std[0],label='std',color='green')

            axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

            title = "Expected result: (c/k)=(" + str(i) + "/" + str(n_expected_clusters) + ")" +  ", N_curves=" + str(N)
            axarr[1].set_title(title)
            plt.show()
            plt.savefig("./plots/" + key_name  + "_expected_time_series__" + str(n_expected_clusters) + "_" + str(i) + ".png")
            plt.close(f)

        summary = self.get_summary_of_solution(expected_solutions,algorithm,dist, n_expected_clusters)

    def get_summary_of_solution(self, expected_solutions,algorithm,dist,k):

        c_solutions = {}
        cluster = self.solutions[algorithm][dist][k]
        for c_id, points in cluster[1].items():
            c_solutions[c_id] = []
            for point in points:
                c_solutions[c_id].append(expected_solutions[point])

        c_summary = {}
        for c_id, vals in c_solutions.items():
            x = np.array(vals)
            y = np.bincount(x.astype('int'))
            ii = np.nonzero(y)[0]
            c_summary[c_id] = np.vstack((ii,y[ii])).T

        return c_summary




