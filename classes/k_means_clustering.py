from classes.clustering import Clustering
from classes.dtw import DTW
from scipy.spatial import distance
from copy import copy

import numpy as np
import colorsys
import sys
import pdb

import matplotlib.pyplot as plt

class K_means_clustering(Clustering):

    def __init__(self, data, k_vector, algorithm_vector, distance_vector,max_iter=10, keep_solutions = False, normalize_by_min=True, normalize_by_max=True):

        super(K_means_clustering,self).__init__(data,k_vector,algorithm_vector, distance_vector, normalize_by_min, normalize_by_max)

        #TODO: ver como deletar isso direito
        #del data[1012]

        self.max_iter = max_iter
        self.keep_solutions = keep_solutions

        self.iter_solutions = {}

        self.best_solution = None
        self.best_fitness = sys.float_info.max

        self.solutions = {}
        self.validations = {}


    def fit(self,verbose=False):
        self.normalize()
        for k in self.k_vector:
            for dist in self.distance_vector:
                for algorithm in self.algorithm_vector:
                    if(verbose):
                        info = "algorithm=%s, k=%s, distance=%s" % (algorithm,k, dist)#
                        print(info)
                    if (algorithm == 'k-means'):
                        self.fit_by_k__means(k,dist,verbose)
                    elif (algorithm == 'k-means++'):
                        self.fit_by_k_means_pp()
                        #TODO: Remover continue apos implementar k-menas++
                        continue

                    self.add_solution(algorithm,dist,k)
                    self.validate_solution(algorithm,dist,k, verbose)

    def add_solution(self,algorithm,dist,k):

        if (algorithm not in self.solutions):
            self.solutions[algorithm] = {}

        if(dist not in self.solutions[algorithm]):
            self.solutions[algorithm][dist] = {}

        self.solutions[algorithm][dist][k] = (self.best_fitness,self.best_solution)

    def validate_solution(self,algorithm,dist,k, verbose):

        #TODO: Implement other validation methods
        self.validate_solution_by_silhouette(algorithm,dist,k, verbose)

    def validate_solution_by_silhouette(self, algorithm, dist, k, verbose):

        s_is = {}
        cluster_data = self.solutions[algorithm][dist][k]
        fitness = cluster_data[0]
        if (verbose):
            print ("Fitness = " + str(fitness))
        cluster = cluster_data[1]
        if (verbose):
            print("Cluster = " + str(cluster))
        centroids = self.get_centroids(dist,cluster)
        if (verbose):
            print("Centroids = " + str(centroids))
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

        self.add_validation(algorithm,dist,k,silhouette)

    def add_validation(self, algorithm,dist,k,s_i):

        if (algorithm not in self.validations):
            self.validations[algorithm] = {}

        if(dist not in self.validations[algorithm]):
            self.validations[algorithm][dist] = {}

        self.validations[algorithm][dist][k] = s_i

    def fit_by_k__means(self,k,dist, verbose):
        for i in range(self.max_iter):
            if (verbose):
                print("Iter: %s/%s"%(i,self.max_iter))
            solution = self.generate_random_solution(k)
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
            #method = getattr(distance,dist)
            fitness = 0
            for key, val in self.data.items():
                min_distance = sys.float_info.max
                best_centroid = -1

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

                fitness += min_distance if min_distance != sys.float_info.max else 0
                if (best_centroid in new_solution):
                    new_solution[best_centroid].append(key)
                elif (best_centroid != -1):
                    new_solution[best_centroid] = list([key])

            iters += 1
            if (verbose):
                print("fitness=" + str(fitness) + ", iter=" + str(iters))
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



    def fit_by_k_means_pp(self):
        print("TODO")

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
        silhouette = self.validations[algorithm][distance][k]
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
            m = np.mean(l,axis=0)
            axarr[0].plot(m[0])
            silhouette = self.validations[algorithm][distance][k]
            title = algorithm + ", d=" + distance + ", (c/k)=(" + str(c) + "/" + str(k) + "), silh.=" + str(silhouette) + ", N_curves=" + str(N)
            axarr[0].set_title(title)
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

    def plot_silhouette(self, key_name):
        fig = plt.figure()
        for alg,d0 in self.solutions.items():
            for dist,d1 in d0.items():
                s = []
                ks = []
                for k,d2 in d1.items():
                    silhouette = self.validations[alg][dist][k]
                    s.append(silhouette)
                    ks.append(k)

                plt.plot(ks,s, label= alg + "," + dist)

        plt.legend(loc="best")
        title = "Silhouettes x K"
        plt.title(title)
        plt.show()

        plt.savefig("./plots/" + key_name + "_silhouette_curves.png")
        plt.close(fig)



