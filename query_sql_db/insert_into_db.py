from preprocessing import ImagePreprocessor
import sqlite3
import os

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")
conn = sqlite3.connect("sql_db.db")

path_to_inserted = "dataset\\Email\\2064207315c.jpg" 
seg = path_to_inserted.split("\\")

filename = str(seg[2])
type = str(seg[1])

process = ImagePreprocessor(path_to_inserted)
description = process.preprocess_and_extract()

command = f"""
Insert into ocr values (?, ?, ?);
"""
cur = conn.cursor()
cur.execute(command, (filename, type, description))
conn.commit()
conn.close()