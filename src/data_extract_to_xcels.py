#######################################################################
# Mihai I 2019
#
# Description: converting the sql export into .xlsx file for further
#development processing.
#
#######################################################################
import MySQLdb
import pandas as pd
import os
#######################################################################
#
# It needs to install the pyodbc, pands and openpyxl library
#
#######################################################################
# Open database connection
db = MySQLdb.connect("localhost","midu","password","mytable" )

sql_select_Query = "select * from mytable"

# prepare a cursor object using cursor() method
cursor = db.cursor()
script = """
SELECT * FROM mytable
"""
#Executing the script commands
cursor.execute(script)

columns = [desc[0] for desc in cursor.description]
data = cursor.fetchall()
df = pd.DataFrame(list(data), columns=columns)

writer = pd.ExcelWriter('C:/Users/Mihai/Documents/GitHub/LTE-Analytics/src/foo.xlsx',options={'encoding':'utf-8'})
df.to_excel(writer,engine='xlsxwriter')
writer.save()
