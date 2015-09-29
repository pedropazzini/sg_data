import numpy as np
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class K_means:


    def __init__(self,X,iterations=100,distance='euclidean',n_clusters=6,mu=np.array([]),print_iter=False):

        self.__mu = None
        self.__n_iter = None
        self.__classification = None
        self.__data = None
        self.__dist_type = None
        self.__k = None
        self.__dic_results = {}
        self.__best_clustering_result = None
        self.__best_fitness = None

        assert (iterations > 1 and mu.size == 0), "mu Parameter necessary..."

        assert (n_clusters > 0), "n_clusters must be greather than 0..."

        total_iter = iterations

        self.__k = n_clusters

        if (mu.size == 0):
            mu = self.__build_mu(X)
        else:
            n_clusters = len(mu)



        self.__data = X
        self.__dist_type = distance

        n = np.shape(X)[0]
        c = self.__k


        while (iterations > 0):

            if(print_iter):
                print(str(total_iter - iterations)+"/"+str(total_iter))

            iterations = iterations -1

            c_sample = []
            c_sample_old = []
            stop_criteria = False
            it = 0
            while(stop_criteria  == False):
                it = it + 1
                for sample in X:
                    d = []
                    for mu_i in mu:
                        d.append(self.get_distance(sample,mu_i,distance))

                    c_sample.append(np.argmin(d))

                mu = self.update_mu(mu,c_sample,X)

                if (np.array_equal(c_sample_old,c_sample)):
                    stop_criteria = True
                    mu = self.__build_mu(X)
                else:
                    c_sample_old = c_sample
                    c_sample = []


            self.__classification = np.array(c_sample)
            self.__validate_cluster_result(iterations)

        self.__mu = mu
        self.__n_iter = it
        self.__classification = self.__best_clustering_result

    def __build_mu(self,data):

        min_v = np.min(data.reshape(-1))
        max_v = np.max(data.reshape(-1))
        vars_ = data.shape[1]
        mu = np.random.uniform(min_v,max_v,self.__k*vars_)
        mu = mu.reshape((self.__k,vars_))


        return mu
        

    def get_mu(self):
        return self.__mu

    def get_iter(self):
        return self.__n_iter

    def get_classification(self):
        return self.__best_clustering_result

    def get_elements_by_class(self):
        d = {}
        for c in np.unique(self.__classification):
            d[c] = self.__data[self.__classification == c]

        return d

    def update_mu(self,mu,c_sample,X):
        new_mu = []
        c = len(mu)
        for i in range(0,c):
            b = (np.array(c_sample) == i)
            div = sum(b)
            num = sum(X[b])
            if (div >0):
                new_mu.append(num/div)
            else:
                new_mu.append(mu[i])

        return np.array(new_mu)

    # Demais métricas de distância podem ser obtidas em http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    def get_distance(self,p1,p2,distancetype):

        method = getattr(distance,distancetype)
        return method(p1,p2)

    def plot_as_time_series(self, ini=None, end=None):
        #cents = self.__best_clustering_result
        cents = range(0,self.__k)
        total = len(cents)
        total_meters = len(self.__data)
        folder = './plots/'
        i = 0
        for c in cents:
            assigns = self.__data[self.__best_clustering_result == c]
            name = "Centroid " + str(i+1) + "/" + str(total) + " ~ " + str(len(assigns)) + "/" + str(total_meters) +  " meters, " + self.__dist_type
            plt.plot(c)
            f, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.plot(c)
            if(ini is not None and end is not None):
                name = name + " - " + str(ini) + " ~ " + str(end)
            ax1.set_title(name)
            for a in assigns:
                #ax2.plot(self.__data[a])
                ax2.plot(a)

            filename = folder + name.replace(' ','_').replace('~','_').replace('/','_') + '.png'
            f.savefig(filename)

            i = i + 1

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c = ['r','b','g']
        m = ['o','*','+']
        d = self.get_elements_by_class()
        k = 0
        for key,val in d.items():
            pts = val.reshape(3,len(val))
            ax.scatter(pts[0],pts[1],pts[2],c = c[key],marker=m[key])
            k = k + 1

        ax.set_label("x1")
        ax.set_label("x2")
        ax.set_label("x3")
        ax.set_title("K-means (k=" + str(k) + ", distance='" + self.__dist_type + "')")

        plt.show()

    def get_result_by_iter(self):
        return self.__dic_results

    def __validate_cluster_result(self,iteration):
        fit = self.__validate_by_sum_square_error()
        self.__dic_results[iteration] = fit

        min_at = min(self.__dic_results, key = self.__dic_results.get)
        if (min_at == iteration):
            self.__best_clustering_result = self.__classification
            self.__best_fitness = fit

    def get_fitness(self):
        return self.__best_fitness


    def __validate_by_sum_square_error(self): 

        dist_method = getattr(distance,self.__dist_type)
        J = 0
        for key, value in self.get_elements_by_class().items():
            m_i = np.mean(value,axis=0)
            sqe = 0
            for x in value:
                sqe = sqe + np.power(dist_method(x,m_i),2)
            J = J + sqe

        return J
