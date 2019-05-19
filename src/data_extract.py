import mysql.connector
from mysql.connector import Error
try:
   mySQLconnection = mysql.connector.connect(host='localhost',
                             database='stats_db',
                             user='username',                   #replace by yourself
                             password='password')               #replace by yourself

   sql_select_Query = "select * from stats_db"
   cursor = mySQLconnection .cursor()
   cursor.execute(sql_select_Query)
   records = cursor.fetchall()

   print("Total number of rows in stats_db is - ", cursor.rowcount)
   print ("Printing each row's column values i.e.  developer record")
   for row in records:
       print("eNodeB = ", row[0], )
       print("Name = ", row[1])
       print("JoiningDate  = ", row[2])
       print("Salary  = ", row[3], "\n")
   cursor.close()

except Error as e :
    print ("Error while connecting to MySQL", e)
finally:
    #closing database connection.
    if(mySQLconnection .is_connected()):
        connection.close()
        print("MySQL connection is closed")
