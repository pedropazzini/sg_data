
from sqlalchemy import Column, Integer, Float
from classes.date import Smart_meter_date
from declarative.classes import Raw_data_dec
from declarative.classes import Meter_data_dec
from declarative.classes import Normalized_measure_dec
from classes.raw_data import Raw_data
from classes.normalized_measure import Normalized_measure
from sqlalchemy.orm import sessionmaker
import numpy as np
from classes.dbconn import Dbconn

odbc = Dbconn()
engine = odbc.get_psql_engine()
DBSession = sessionmaker(bind=engine)

class Meter_data():

    __id_meter_data = -1
    __id_meter = -1
    __id_meter_data_collection = -1

    __measurements = np.array([])
    __normalized_measurements = np.array([])
    __dates = []

    def __init__(self,id_meter, ini, end, id_meter_data_collection = None, insert=False):

        if(id_meter_data_collection is not None):
            self.__id_meter_data_collection = id_meter_data_collection        

        self.__id_meter = id_meter
        
        if(insert):
            self.insert()


        r = Raw_data()
        v = []
        d = []
        for item in r.get_interval(ini,end,id_meter):            
            v.append(item.kwh)
            d.append(item.date)

        self.__measurements = np.array(v)
        self.__dates = d

    def get_min_max(self, normalized=False):
        if (normalized):
            if (len(self.__normalized_measurements) > 0):
                return min(self.__normalized_measurements),max(self.__normalized_measurements)
            else:
                return 10e5,-10e5
        else:
            if (len(self.__measurements) > 0):
                return min(self.__measurements),max(self.__measurements)
            else:
                return 10e5,-10e5
        
    def normalize(self,min_val,max_val):
        self.__normalized_measurements = (self.__measurements-min_val)/(max_val-min_val)

    def insert(self):
    
        ## Insert the meter_data on the data base
        md_obj = Meter_data_dec()
        md_obj.id_meter = self.__id_meter
        md_obj.id_meter_data_collection = self.__id_meter_data_collection
        session = DBSession()
        try:
            session.add(md_obj)
            session.commit()
            session.refresh(md_obj)
            self.__id_meter_data = md_obj.id_meter_data
        except:
            session.rollback()
            raise
        session.close()

        ## Insert each measurement of the meter_data
        i = 0
        for m in self.__normalized_measurements:
            nm = Normalized_measure(self.__id_meter_data,m,self.__dates[i])
            nm.insert()
            i = i + 1

    def get_normalized_measures(self):
        session = DBSession()
        r = session.query(Normalized_measure_dec).filter(Normalized_measure_dec.id_meter_data == self.__id_meter_data).all()
        session.close()
        return r

    def get_normalized_measures(id_meter_data):
        session = DBSession()
        r = session.query(Normalized_measure_dec).filter(Normalized_measure_dec.id_meter_data == id_meter_data).all()
        session.close()
        return r
