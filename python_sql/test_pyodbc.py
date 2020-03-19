import pyodbc
import pandas as pd

print(pyodbc.drivers())
# ['SQL Server', 'SQL Server Native Client 11.0', 'SQL Server Native Client RDA 11.0', 'ODBC Driver 13 for SQL Server']
driver = 'ODBC Driver 17 for SQL Server'
server = r'DESKTOP-YOGA\SQLEXPRESS'
database = 'test'
con_string = 'DRIVER={};SERVER={};DATABASE={};Trusted_Connection=yes'.format(driver, server, database)  #UID=user;PWD=password
print(con_string)
con = pyodbc.connect(con_string)
df = pd.read_sql('SELECT TOP 10 * FROM {}'.format('usertable'), con)
print(df.shape)