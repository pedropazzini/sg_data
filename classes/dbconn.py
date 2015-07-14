import configparser
from peewee import *
from sqlalchemy import create_engine

class Dbconn:

    __psql_db = None 
    __psql_engine = None
    __psql_engine_str = None


    def __init__ (self):
        config = configparser.ConfigParser()
        config.read('./plot/sg.cfg')

        self.__psql_db = PostgresqlDatabase(config.get('Database','db'),user=config.get('Database','user'),password=config.get('Database','password'),host=config.get('Database','ip'),port=config.get('Database','port'))

        self.__psql_engine_str = 'postgresql+psycopg2://' + config.get('Database','user') + ':' + config.get('Database','password') + '@' + config.get('Database','ip') + ':' + config.get('Database','port') + '/' + config.get('Database','db')

    def get_conn(self):
        return self.__psql_db

    def get_psql_engine_str(self):
        return self.__psql_engine_str

    def get_psql_engine(self):
        return create_engine(self.get_psql_engine_str())


        
