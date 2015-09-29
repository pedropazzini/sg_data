from sqlalchemy import Column, Integer, Float
from classes.date import Smart_meter_date
from declarative.classes import Raw_data_dec
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from classes.dbconn import Dbconn

import sys
import copy
import datetime
import numpy as np

odbc = Dbconn()
engine = odbc.get_psql_engine()

DBSession = sessionmaker(bind=engine)
session = DBSession()

class Raw_data():

    __max_none_counts = 4

    def __init__(self,id_meter=-1,date=-1,kwh=-1,d_time = None):

        self.id_meter = id_meter
        self.date = date
        self.kwh = kwh

        self.smart_meter_date = Smart_meter_date(self.date)

        self.d_time = None

        if (d_time is None):
            self.d_time = self.smart_meter_date.get_datetime()
            self.update()
        elif (d_time != self.smart_meter_date.get_datetime()):
            raise ValueError("Different times on database. Incorrect values for " + self.print())
        else:
            self.d_time = d_time


    def __repr__(self):
        return "<Raw_data(id_meter='%s', date='%s', kwh='%s', d_time='%s')>" % (self.id_meter, self.date, self.kwh, self.d_time)

    def get_interval (self,ini,end,meter_id):
        smd_ini = Smart_meter_date()
        smd_ini.set_datetime(ini.year,ini.month,ini.day,ini.hour,ini.minute)
        
        smd_end = Smart_meter_date()
        smd_end.set_datetime(end.year,end.month,end.day,end.hour,end.minute)

        r = Raw_data_dec()

        try:
            return session.query(Raw_data_dec).filter(Raw_data_dec.id_meter == meter_id, Raw_data_dec.date >= smd_ini.get_integer_time(), Raw_data_dec.date <= smd_end.get_integer_time()).all()
        except:
            session.rollback()
        finally:
            session.close()

    def get_full_data(self, arraysize = 1000):
        try:
            offset = 0
            while True:
                  r = False
                  for elem in session.query(Raw_data_dec).limit(arraysize).offset(offset):
                      if (elem.d_time is not None):
                          continue
                      r = True
                      rd = Raw_data(elem.id_meter,elem.date,elem.kwh)
                      yield rd
                      offset += 1000
                      if not r:
                          break
        except:
            session.rollback()
            print("Exception getting full data: " + str(sys.exc_info()[0]))
        finally:
            session.close()

    def get_id_meters(self):
        try:
            results = session.query(distinct(Raw_data_dec.id_meter)).all()
            vals, = zip(*results)
            return vals
        except:
            session.rollback()
        finally:
            session.close()

    def update(self):
        session = DBSession()
        try:
            session.query(Raw_data_dec).filter(Raw_data_dec.id_meter == self.id_meter, Raw_data_dec.date == self.date).update({'d_time':self.d_time})
            session.commit()
        except:
            session.rollback()
            print("Exception on UPDATE: " + str(sys.exc_info()[0]))
            print("Exception on UPDATE: " + str(sys.exc_info()[1]))
        finally:
            session.close()

    def get(self):
        
        if (self.id_meter is None or self.id_meter < 1):
            raise ValueError("'id_meter' not set on get() methos")

        if (self.date is None or self.date < 1):
            raise ValueError("'date' not set on get() methos")

        session = DBSession()
        try:
            result = session.query(Raw_data_dec).filter(Raw_data_dec.id_meter == self.id_meter, Raw_data_dec.date == self.date).first()
            if (result is None):
                print("Not found: id_meter[" + str(self.id_meter) + "], date[" + str(self.date) + "].")
                return None
            else:
                return Raw_data(result.id_meter,result.date,result.kwh,result.d_time)
        except:
            session.rollback()
            print("Exception on raw_data.get(): " + str(sys.exc_info()[0]))
            print("Exception on raw_data.get(): " + str(sys.exc_info()[1]))
        finally:
            session.close()

    def add_time_delta(self,time_delta):
        self.smart_meter_date.add_time_delta(time_delta)
        self.date = self.smart_meter_date.get_integer_time()
        self.d_time = self.smart_meter_date.get_datetime()

        new_raw_data = self.get()
        if (new_raw_data is None):
            self.kwh = None
        else:
            self = new_raw_data

    def exists(self):
        return self.kwh is None

    def get_profile(self, smd_ini_date, smd_end_date, is_weekdays = False):

        prof = self.get_rough_profile(smd_ini_date,smd_end_date,is_weekdays)

        l_prof = None if prof is None else self.lapidate_profile(prof)

        return None if l_prof is None else self.normalize_profile(l_prof)

    def normalize_profile(self, prof):

        new_prof = {}
        new_prof['S']={}
        new_prof['W']={}
        new_prof['T']={}

        for season,data in prof.items():
            for time, measures in data.items():
                new_prof[season][time] = np.mean(measures)

        for season, data in new_prof.items():
            a = np.array(list(data.values()))
            min_a = np.min(a)
            max_a = np.max(a)
            for time, measures in data.items():
                new_prof[season][time] = (max_a - new_prof[season][time])/(max_a-min_a)

        return new_prof

    def lapidate_profile(self, profile):

        for season, data in profile.items():
            for time, measures  in data.items():
                a = np.array(measures)
                none = -1
                none_count = 0
                i = 0
                length = len(a)
                for m in a:
                    if m == -1:
                        none_count += 1
                        if (none_count >= self.__max_none_counts):
                            print("Quitting becouse: " + str(none_count))
                            return None
                    else:
                        if (none_count != 0):
                            init = i - none_count - 1 if i - none_count - 1 >= 0 else i
                            end = i
                            quant = (a[end] - a[init])/(none_count+1)
                            for j in range(none_count):
                                a[init+j+1] = a[init] + (j+1)*quant
                                measures[init+j+1] = measures[init] + (j+1)*quant
                        none_count = 0

                    i += 1

        return profile




    def get_rough_profile(self, smd_ini_date, smd_end_date, is_weekdays = False):


        profile = {}
        profile['S'] ={} # summer
        profile['W'] ={} # winter
        profile['T'] ={} # transition (spring and autumn)

        current_time = copy.copy(smd_ini_date)
        rd = Raw_data(id_meter=self.id_meter,date=current_time.get_integer_time())

        none_counts = 0
        while (current_time <= smd_end_date):
            if ( current_time.is_day_of_week() != is_weekdays):
                new_rd = rd.get()
                measure_kwh = -1 # None measure
                if (new_rd is not None):
                    measure_kwh = new_rd.kwh
                    found = True
                    none_counts = none_counts - 1 if none_counts > 0 else 0
                else:
                    none_counts += 1
                    if(none_counts > self.__max_none_counts):
                        print("Quitting on roughbecouse: " + str(none_counts))
                        return None
                str_key = 'S' if current_time.is_summer() else ('W' if current_time.is_winter() else 'T')
                dt = current_time.get_datetime()
                time = str(dt.hour) + ":" + str(dt.minute)
                if (time not in profile[str_key]):
                   profile[str_key][time] = []
                profile[str_key][time].append(measure_kwh)

                rd.add_time_delta(datetime.timedelta(minutes=30))
                current_time.set_datetime(rd.d_time.year,rd.d_time.month,rd.d_time.day,rd.d_time.hour,rd.d_time.minute)

            else:
                dow = current_time.get_datetime().weekday()
                pivot = 5 if is_weekdays else 7
                diff = pivot-dow
                rd.add_time_delta(datetime.timedelta(days=diff))
                current_time.set_datetime(rd.d_time.year,rd.d_time.month,rd.d_time.day,0,0)

        return profile



