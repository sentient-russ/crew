# Dependencies 
# python -m pip install mysql-connector-python
import mysql.connector
import os
import constants as cnst

mydb = mysql.connector.connect(
  host = cnst.db_server,
  user = cnst.db_user,
  password = cnst.db_password
)
 