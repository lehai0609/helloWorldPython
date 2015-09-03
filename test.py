import sqlite3 as lite
import pandas as pd
con = lite.connect('getting_started.db')
with con:
    cur = con.cursor()
    sql = "SELECT name, state, year, warmth_month, cold_month FROM cities " \
          "INNER JOIN weather " \
          "ON name = city"
    cur.execute(sql)

rows = cur.fetchall()
cols = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows, columns=cols)
# print(df)
for index, row in df.iterrows():
    print("In %s the warmest month is %s" %(row['name'],row['warmth_month']))