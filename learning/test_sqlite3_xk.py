# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:52:52 2017

@author: Xianhui
"""

# test sqlite3
import sqlite3
#%%
conn = sqlite3.connect("example.db")
cur = conn.cursor()
cur.execute('''DROP TABLE IF EXISTS stocks''')
cur.execute('''CREATE TABLE stocks 
            (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
            date text, price real)''')
cur.execute('''INSERT OR IGNORE INTO  stocks VALUES(1, '12/2017', 1.1)''')
cur.execute('''INSERT OR IGNORE INTO  stocks VALUES(1, '12/2017', 1.1)''')
cur.execute('''INSERT OR IGNORE INTO  stocks VALUES(2, '12/2017', 1.1)''')
cur.execute('''SELECT * FROM stocks''')
conn.commit()
print(cur.fetchall())
conn.close()
