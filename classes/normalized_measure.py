from sqlalchemy.orm import sessionmaker
from declarative.classes import Normalized_measure_dec
from classes.dbconn import Dbconn

odbc = Dbconn()
engine = odbc.get_psql_engine()

DBSession = sessionmaker(bind=engine)

class Normalized_measure:

    __id_normalized_measure = -1
    __id_meter_data = -1
    __normalized_measure = -1
    __date = -1

    def __init__(self,id_meter_data,normalized_measure, date, id_normalized_measure = None):

        self.__id_meter_data = id_meter_data
        self.__normalized_measure = normalized_measure
        self.__date = date
        self.__id_normalized_measure = id_normalized_measure

    def insert(self):

        session = DBSession()
        nm_obj = Normalized_measure_dec()
        nm_obj.id_meter_data = self.__id_meter_data
        nm_obj.normalized_measure = self.__normalized_measure
        nm_obj.date = self.__date

        session.add(nm_obj)
        session.commit()
        session.refresh(nm_obj)
        self.__id_normalized_measure = nm_obj.id_normalized_measure
        session.close()


