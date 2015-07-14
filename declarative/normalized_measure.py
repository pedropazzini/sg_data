
from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Normalized_measure_dec(Base):

    __tablename__ = 'normalized_measure'

    id_meter_data = Column(Integer, primary_key=True)
    normalized_measure = Column(Float, nullable=False)

    def __repr__(self):
        return "<Normalized_measure (id_meter_data='%s', normalized_measure='%s')>" % (self.id_meter_data, self.normalized_measure)

