from datetime import datetime
import numpy as np
from classes.row_data import Row_data
from classes.meter_data_collection import Meter_data_collection
from classes.ts_cluster import ts_cluster

ini = datetime(2010,5,10,0,0)
end = datetime(2010,5,11,0,0)

m = list(range(1001,1004,1))

mdc_obj = Meter_data_collection(ini,end,m)
#mdc_obj = Meter_data_collection(ini,end)

#mdc_obj.cluster_k_means(6,8,300,True,True,True,distance='all')
#mdc_obj.cluster_k_means(22,8,300,True,True,True)
