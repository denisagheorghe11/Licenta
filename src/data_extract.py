#!/usr/bin/python
###############################################
#
#
# Description: Python3.7 Script to access and
#retrieve mysqlDB parameters in LTEAnalytics.
#
# Mihai I
###############################################
import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","midu","password","mytable" )

sql_select_Query = "select * from mytable"

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# execute another SQL query
cursor.execute(sql_select_Query)

# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print( "Database version : %s ",data)

records = cursor.fetchall()
print("Total number of rows in mytable is - ", cursor.rowcount)

# for testing
print ("Print each row's column value!")
for row in records:
       print("bs_id = ", row[0], )
       print("agent_info0agent_id = ", row[1])
       print("agent_info0bs_id  = ", row[2])
       print("agent_info0capabilities0  = ", row[3], "\n")
cursor.close()

# disconnect from server
db.close()
