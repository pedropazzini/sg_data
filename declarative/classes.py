from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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


class Meter_data_dec(Base):

    __tablename__ = 'meter_data'

    id_meter_data = Column(Integer, primary_key=True)
    id_meter = Column(Integer, nullable=False)
    id_meter_data_collection = Column(ForeignKey("meter_data_collection.id_meter_data_collection"), nullable=False)

    normalized_measure_dec = relationship("Normalized_measure_dec")


    def __repr__(self):
        return "<Meter_data (id_meter_data='%s', id_meter='%s', id_meter_data_collection='%s')>" % (self.id_meter_data, self.id_meter, self.id_meter_data_collection)

class Raw_data_dec(Base):

    __tablename__ = 'raw_data'

    id_meter = Column(Integer, primary_key=True)
    date = Column(Integer, primary_key=True)
    d_time = Column(DateTime)
    kwh = Column(Float)

    def __repr__(self):
        return "<Raw_data(id_meter='%s', date='%s', kwh='%s', d_time='%s')>" % (self.id_meter, self.date, self.kwh, self.d_time)

class Normalized_measure_dec(Base):

    __tablename__ = 'normalized_measure'

    id_normalized_measure = Column(Integer, primary_key = True)
    id_meter_data = Column(ForeignKey("meter_data.id_meter_data"))
    normalized_measure = Column(Float, nullable=False)
    date = Column(Integer,nullable=False)

    def __repr__(self):
        return "<Normalized_measure (id_normalized_measure='%s', id_meter_data='%s', normalized_measure='%s', date='%s')>" % (self.id_normalized_measure, self.id_meter_data, self.normalized_measure, self.date)

