from sqlalchemy import Column, Integer, Float
from classes.date import Smart_meter_date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Row_data_dec(Base):

    __tablename__ = 'row_data'

    id_meter = Column(Integer, primary_key=True)
    date = Column(Integer, primary_key=True)
    kwh = Column(Float)

    def __repr__(self):
        return "<Row_data(id_meter='%s', date='%s', kwh='%s')>" % (self.id_meter, self.date, self.kwh)

