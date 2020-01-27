import mysql.connector as mysql
import pandas as pd


class MysqlBackendConnector:
    def __init__(self, dbname, user, password):
        self.db = mysql.connect(
            host="localhost", user=user, passwd=password, database=dbname)

    def signup(self, email, password):
        query = "SELECT email FROM authentication where email=%s"
        cursor = self.db.cursor()
        values = (email,)
        cursor.execute(query, values)
        cursor.fetchall()
        print(cursor.rowcount)
        if cursor.rowcount == 0:
            query = "INSERT INTO authentication(email,password) VALUES(%s,%s)"
            values = (email, password)
            cursor.execute(query, values)
            self.db.commit()
        else:
            print("User has already signed up")

    def login(self, email, password):
        query = "SELECT password FROM authentication where email=%s"
        cursor = self.db.cursor()
        values = (email,)
        cursor.execute(query, values)
        results = cursor.fetchall()
        if cursor.rowcount != 0:
            for pass_ in results:
                if password == pass_[0]:
                    return True
        return False
