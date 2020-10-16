import sqlalchemy as sql
from urllib.parse import quote_plus as urlquote
#import cx_Oracle
import pymysql
import psycopg2
import sqlalchemy_redshift

db_tool = {'mysql': 'mysql+pymysql',
           'redshift': 'redshift+psycopg2',
           'oracle': 'oracle+cx_oracle'}


def db_connection(connection_details):
    connection_strings = {'db': db_tool[connection_details[0]],
                          'user': urlquote(connection_details[1]),
                          'pwd': urlquote(connection_details[2]),
                          'host': connection_details[3],
                          'name':  connection_details[4]}
    eng_str = "{db}://{user}:{pwd}@{host}/{name}".format(**connection_strings)
    if db_tool[connection_details[0]] == 'redshift':
        engine = sql.create_engine(eng_str,
                                   connect_args={'sslmode': 'disable'})
    else:
        engine = sql.create_engine(eng_str)
    return engine
