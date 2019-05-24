import mysql.connector
from mysql.connector import Error
try:
   mySQLconnection = mysql.connector.connect(host='localhost',
                             database='stats_db',
                             user='midu',                   #replace by yourself
                             password='password')               #replace by yourself

   sql_select_Query = "select * from stats_db"
   cursor = mySQLconnection .cursor()
   cursor.execute(sql_select_Query)
   records = cursor.fetchall()

   print("Total number of rows in stats_db is - ", cursor.rowcount)
   print ("Printing each row's column values i.e.  developer record!This is just for testing")
   for row in records:
       print("bs_id = ", row[0], )
       print("agent_info0agent_id = ", row[1])
       print("agent_info0bs_id  = ", row[2])
       print("agent_info0capabilities0  = ", row[3], "\n")
   cursor.close()

except Error as e :
    print ("Error while connecting to MySQL", e)
finally:
    #closing database connection.
    if(mySQLconnection .is_connected()):
        connection.close()
        print("MySQL connection is closed")
