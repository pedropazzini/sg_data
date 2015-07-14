from sqlalchemy import Column, Integer, Float
from classes.date import Smart_meter_date
from declarative.classes import Row_data_dec
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from classes.dbconn import Dbconn

odbc = Dbconn()
engine = odbc.get_psql_engine()

DBSession = sessionmaker(bind=engine)
session = DBSession()

class Row_data():

    __id_meter = -1
    __date = -1
    __kwh = -1

    __smart_meter_date = None

    def __init__(self,id_meter=-1,date=-1,kwh=-1):

        self.__id_meter = id_meter
        self.__date = date
        self.__kwh = kwh

        self.__smart_meter_date = Smart_meter_date(self.__date)


    def print(self):
        return "<Row_data(id_meter='%s', date='%s', kwh='%s')>" % (self.__id_meter, self.__date, self.__kwh)

    def get_interval (self,ini,end,meter_id):
        smd_ini = Smart_meter_date()
        smd_ini.set_datetime(ini.year,ini.month,ini.day,ini.hour,ini.minute)
        
        smd_end = Smart_meter_date()
        smd_end.set_datetime(end.year,end.month,end.day,end.hour,end.minute)

        r = Row_data_dec()

        return session.query(Row_data_dec).filter(Row_data_dec.id_meter == meter_id, Row_data_dec.date >= smd_ini.get_integer_time(), Row_data_dec.date <= smd_end.get_integer_time()).all()

    def get_id_meters():
        results = session.query(distinct(Row_data_dec.id_meter)).all()
        vals, = zip(*results)
        return vals

