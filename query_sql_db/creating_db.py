import sqlite3
import os
conn = sqlite3.connect("sql_db.db")
os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")

cur = conn.cursor()
command = f"""
create table ocr (
filename varchar primary key,
type varchar,
description text
);
"""

cur.execute(command)
conn.close()

