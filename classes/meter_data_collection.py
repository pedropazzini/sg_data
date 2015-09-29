from sqlalchemy import Column, Integer, Float
from classes.date import Smart_meter_date
from declarative.classes import Raw_data_dec
from declarative.classes import Meter_data_dec
from classes.meter_data import Meter_data
from classes.raw_data import Raw_data
from declarative.classes import Meter_data_collection_dec
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from classes.ts_cluster import ts_cluster
from classes.k_means import K_means
from classes.dbconn import Dbconn

odbc = Dbconn()
engine = odbc.get_psql_engine()
DBSession = sessionmaker(bind=engine)
        
class Meter_data_collection():

    __id_meter_data_collection = None
    __ini_date = None
    __end_date = None
    __min_val = 10e5
    __max_val = -10e5

    __meters_data = {}

    def __init__(self,ini_date, end_date, id_meters = None):

        self.__ini_date = ini_date
        self.__end_date = end_date

        ids = []
        if id_meters is None:
            ids = Raw_data.get_id_meters()
        else:
            ids = id_meters

        self.insert_update()

        # Creates each object and find mininma maxima
        for meter in ids:
            meter_data_obj = Meter_data(meter,self.__ini_date,self.__end_date,self.__id_meter_data_collection)
            min_obj,max_obj = meter_data_obj.get_min_max()

            if (min_obj < self.__min_val):
                self.__min_val = min_obj
            if(max_obj > self.__max_val):
                self.__max_val = max_obj

            self.__meters_data[meter] = meter_data_obj
        
        # after finding min and max, update
        self.insert_update()

        # Normalize all the measurements and insert into DB
        for meter_id,meter_data_obj in self.__meters_data.items():
            meter_data_obj.normalize(self.__min_val,self.__max_val)
            meter_data_obj.insert()


    def insert_update(self):

        mdc_obj = Meter_data_collection_dec()
        mdc_obj.ini_date = self.__ini_date
        mdc_obj.end_date = self.__end_date

        if(self.__max_val  > -10e4):
            mdc_obj.max_val = self.__max_val

        if(self.__min_val < 10e4):
            mdc_obj.min_val = self.__min_val
        

        session = DBSession()
        try:
            if (self.__id_meter_data_collection is None):
                session.add(mdc_obj)
                session.commit()
                session.refresh(mdc_obj)
                self.__id_meter_data_collection = mdc_obj.id_meter_data_collection
            else:
                raise
        except:
            session.rollback()
            unique_obj = Meter_data_collection_dec()
            obj = session.query(Meter_data_collection_dec).get(self.__id_meter_data_collection)
            obj.max_val = self.__max_val
            obj.min_val = self.__min_val
            session.commit()
        finally:
            session.close()

    def get_meters_data(self):
        session = DBSession()
        result = session.query(Meter_data_dec).filter(Meter_data_dec.id_meter_data_collection == self.__id_meter_data_collection).all()
        session.close()
        return result

    def get_full_data(self, cleaned_data = False):
        dic = {}
        for meter_data in self.get_meters_data():
            measurements = []
            for normalized_measure in Meter_data.get_normalized_measures(meter_data.id_meter_data):
                measurements.append(normalized_measure.normalized_measure)
            meter_vals = np.array(measurements)
            dic[meter_data.id_meter] = meter_vals


        #clean the data
        if (cleaned_data):
            to_remove = []
            for k,v in dic.items():
                if(len(v) == 0):
                    to_remove.append(k)
            for d in to_remove:
                del dic[d]

        return dic


    def plot(self, name_plot = None, locator = "Day"):

        x = []
        y = []

        fig = plt.figure()
        fig.suptitle("Cluster (" + str(self.__ini_date) + " ~ " + str(self.__end_date) + ")" ,fontsize=14,fontweight='bold')

        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_ylabel("kwh ("+str(self.__min_val)+" ~ "+str(self.__max_val)+")")

        string = ''
        legends = []
        #for id_meter,meter_data in self.__m_ids_meters.items():
        for md_obj in self.get_meters_data():
            for nm_obj in Meter_data.get_normalized_measures(md_obj.id_meter_data):
                smd_obj = Smart_meter_date(nm_obj.date)
                x.append(smd_obj.get_datetime())
                y.append(nm_obj.normalized_measure)

            ax.plot(x,y)
            string = string + str(md_obj.id_meter) + '_'
            x = []
            y = []

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H'))
        if (locator == "Day"):
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        else:
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=4))
        plt.gcf().autofmt_xdate()

        filename = string if name_plot is None else name_plot

        filename = filename + '.png'

        fig.savefig(filename)

    def cluster_k_means(self, n_cluster, window_size, n_iter,print_iter=False, plot_profile_load = False, plot_clusters=False, distance='dtw'):

        data = self.get_full_data(True)
        measures = np.vstack(data.values())

        if (distance == 'dtw'):
            tc_obj = ts_cluster(n_cluster)
            tc_obj.k_means_clust(list(measures),n_iter,window_size,print_iter)
            if(plot_profile_load):
                tc_obj.plot_centroids()
            if(plot_clusters):
                tc_obj.plot_centroids_and_assignments(measures,self.__ini_date,self.__end_date)
        else:
            ## TODO: Complete other distances, availabel at http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
            #dist_types = ['braycurtis','canberra','chebyshev','cityblock','correlation','cosine','dice','euclidean','hamming','jaccard','kulsinski','mahalanobis','matching'] 
            dist_types = ['braycurtis','canberra','chebyshev','cityblock','correlation','cosine','euclidean','hamming','jaccard','matching'] 
            results = {}
            for d in dist_types:
                kd = K_means(measures,distance=d)
                fit = kd.get_fitness()
                results[d] = fit
                kd.plot_as_time_series(self.__ini_date, self.__end_date)

            print(results)
