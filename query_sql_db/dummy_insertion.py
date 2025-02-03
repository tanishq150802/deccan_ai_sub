import sqlite3
conn = sqlite3.connect("sql_db.db")

cur = conn.cursor()
command = f"""
Insert into ocr values ("dummy.jpg", "dummy_type", "This is a dummy entry into the db.");
"""

cur.execute(command)
conn.commit()
conn.close()