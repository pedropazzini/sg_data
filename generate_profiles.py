from classes.raw_data import Raw_data
from classes.date import Smart_meter_date

import datetime
import copy

rd = Raw_data()
meters = rd.get_id_meters()

smd_ini_date = Smart_meter_date()
smd_ini_date.set_datetime(year=2009,month=12,day=21,hour=0,minute=0)

smd_end_date = Smart_meter_date()
smd_end_date.set_datetime(year=2010,month=12,day=21,hour=0,minute=0)
i = 0
weekdays = list(range(0,6))
d ={}
total_meters = 0
for meter in meters:
    rd = Raw_data(id_meter=meter)
    print(meter)
    d[meter] = rd.get_profile(smd_ini_date,smd_end_date)
    print(d[meter])
    total_meters += 1
    if (total_meters > 3000):
        break

