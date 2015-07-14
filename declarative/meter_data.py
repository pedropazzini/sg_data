from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
#from declarative.meter_data_collection import Meter_data_collection_dec

Base = declarative_base()

class Meter_data_dec(Base):

    __tablename__ = 'meter_data'

    id_meter_data = Column(Integer, primary_key=True)
    id_meter = Column(Integer, nullable=False)
    id_meter_data_collection = Column(ForeignKey("meter_data_collection.id_meter_data_collection"), nullable=False)


    def __repr__(self):
        return "<Meter_data (id_meter_data='%s', id_meter='%s', id_meter_data_collection='%s')>" % (self.id_meter_data, self.id_meter, self.id_meter_data_collection)

