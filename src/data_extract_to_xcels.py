#######################################################################
# Mihai I 2019
#
################################################################################
import os
import MySQLdb as dbapi                   # pip install mysqldb
import pandas as pd                       # pip install pandas

cd = os.path.dirname(os.path.abspath(__file__))

# OPEN DATABASE CONNECTION
db = dbapi.connect(host='localhost',user='username',passwd='password', db='mytable')
cur = db.cursor()

# OBTAIN ALL TABLES
cur.execute("SHOW TABLES;")
tables = cur.fetchall()

for t in tables:
    columns = []
    # IMPORT DATA TO DATA FRAME
    df = pd.read_sql("SELECT * FROM {0};".format(t[0]), db)
    # EXPORT DATA FRAME TO CSV
    df.to_csv(os.path.join(cd, '{0}.csv'.format(t[0])), index=False)
# CLOSE CURSOR AND DATABASE CONNECTION
cur.close()
db.close()
################################################################################
###################################EOF##########################################
