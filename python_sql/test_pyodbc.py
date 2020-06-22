import pyodbc
import sqlalchemy
import numpy as np
import pandas as pd

print(pyodbc.drivers())
# ['SQL Server', 'SQL Server Native Client 11.0', 'SQL Server Native Client RDA 11.0', 'ODBC Driver 13 for SQL Server']
driver = 'SQL Server'
server = r'DESKTOP-YOGA\SQLEXPRESS'
database = 'test'
con_string = 'DRIVER={};SERVER={};DATABASE={};Trusted_Connection=yes'.format(driver, server, database)  #UID=user;PWD=password
print(con_string)
with pyodbc.connect(con_string) as con: 
    df = pd.read_sql('SELECT TOP 10 * FROM {}'.format('usertable'), con)
print(df.shape, df.info(), df)
DB = {'servername': server,
      'database': database,
      'driver': 'driver=SQL Server'}
    #   'driver': 'driver=SQL Server Native Client 11.0'}
conn = sqlalchemy.create_engine('mssql+pyodbc://' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'])
df2 = pd.DataFrame(np.array(range(20)).reshape((5, 4)), columns='user_id, useract, actdate, acttime'.split(', '))
print(df2)
df2.to_sql('usertable', con=conn, if_exists='append', index=False)
with pyodbc.connect(con_string) as con: 
    df3 = pd.read_sql('SELECT TOP 10 * FROM {}'.format('usertable'), con)
print(df3)