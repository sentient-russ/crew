# Dependencies 
# python -m pip install mysql-connector-python
import mysql.connector
import os
import constants as cnst

mydb = mysql.connector.connect(
  host = cnst.db_server,
  user = cnst.db_user,
  password = cnst.db_password,
  database = cnst.db_name
)
 
def store_reflection(
    ref_title,
    ref_date,
    ref_passage,
    passage_source,
    experience_p1,
    experience_p2,
    experience_p3,
    experience_p4,
    passage_topic,
    passage_summery,
    wild_rumpus
    ):
    csr = mydb.cursor()
    sql = "INSERT INTO reflections (title, date, passage, source, p1, p2, p3, p4, passage_topic, passage_summery, wild_rumpus) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = (ref_title, ref_date, ref_passage, passage_source, experience_p1, experience_p2, 
           experience_p3, experience_p4, passage_topic, passage_summery, wild_rumpus)
    csr.execute(sql,val)
    mydb.commit()