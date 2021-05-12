# Simple database wrapper class only implementing basic functionality
# Do not use in production ;-)

import sqlite3

SQL_SQLITE_DB_PATH = 'src/resources/sql/results.sqlite3'

SQL_RESULTS_MULTI_INSERT = '''
            INSERT OR IGNORE INTO results 
            (test_dataset, train_dataset, iteration, train_size, gamma, attention_func,
                sentence, y_true, y_pred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''


class ResultsRepository:
    instance = None

    def __init__(self):
        if ResultsRepository.instance is not None:
            # Break if an instance of this class already exists
            raise Exception('Another instance of the Repository Class exists.')

        ResultsRepository.instance = self

        # Just keep the connection open all the time
        self.con = self.connect()
        self.cur = self.con.cursor()

    def __del__(self):
        self.disconnect()

    def connect(self):
         return sqlite3.connect(SQL_SQLITE_DB_PATH)

    def disconnect(self):
        self.con.close()

    def bulk_insert(self, rows):
        self.cur.executemany(SQL_RESULTS_MULTI_INSERT, rows)
        self.con.commit()
