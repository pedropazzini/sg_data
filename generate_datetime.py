from classes.raw_data import Raw_data

rd = Raw_data()

for raw_data_obj in rd.get_full_data():
    print(raw_data_obj)
    #raw_data_obj.update()


