from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from declarative.meter_data import Meter_data_dec

Base = declarative_base()

class Meter_data_collection_dec(Base):

    __tablename__ = 'meter_data_collection'

    id_meter_data_collection = Column(Integer, primary_key=True)
    ini_date = Column(DateTime)
    end_date = Column(DateTime)
    min_val = Column(Float)
    max_val = Column(Float)

    meter_data_collection = relationship("Meter_data_dec")

    def __repr__(self):
        return "<Meter_data_collection (id_meter_data_collection='%s', ini_date='%s', end_date='%s', min_val='%s', max_val='%s')>" % (self.id_meter_data_collection, self.ini_date, self.end_date, self.min_val, self.max_val)

