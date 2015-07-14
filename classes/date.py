
from datetime import datetime

class Smart_meter_date:
    """ Class that conversts the smart meter date to real date and vice and versa """
    m_min_datetime = datetime(2009,1,1,0,0,0)
    m_min_timestamp = m_min_datetime.timestamp()

    m_max_datetime = datetime(2012,1,1,0,0,0)
    m_max_timestamp = m_min_datetime.timestamp()

    def __init__(self,int_date= 10001):
        self.m_integer_date = int_date
        self.update_datetime_from_int()

#    def set_datetime(self,new_datetime):
 #       self.set_datetime(new_dateetime.year,new_datetime.month,new_datetime.day,new_datetime.minute,new_datetime.second)

    def set_datetime(self,year,month,day,hour,minute):
        if (self.m_min_datetime.year > year):
            raise NameError(print ("Onvalid year value:",year,". The year should be greather or equal to ",self.m_min_datetime.year))
        else:
            self.m_real_datetime = datetime(year,month,day,hour,minute)            
            self.update_int_from_datetime()


    def update_datetime_from_int(self):
        days = (self.m_integer_date//100)
        minutes = self.m_integer_date - days*100
        timestamp = self.m_min_timestamp + days*3600*24 + minutes*30*60
        self.m_real_datetime = datetime.fromtimestamp(timestamp)

    def update_int_from_datetime(self):
        base = self.m_real_datetime-self.m_min_datetime
        days = base.days
        seconds = base.seconds
        share_seconds = seconds//60//30
        self.m_integer_date = days*100+share_seconds

    def is_day_of_week(self):
        return self.m_real_datetime.weekday() < 5
    # The function weekday() return the day of the week as an integer, where Monday is 0 and Sunday is 6.

    def get_datetime(self):
        return self.m_real_datetime

    def get_integer_time(self):
        return self.m_integer_date


