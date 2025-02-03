from fastapi import FastAPI
import os
import warnings
import sqlite3
warnings.filterwarnings('ignore')
from model import genai_model
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
google_api_key = os.getenv("GOOGLE_API_KEY")
model = genai_model()

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")

conn = sqlite3.connect("sql_db.db")
cur = conn.cursor()

app = FastAPI()

@app.post("/query_file/")
async def uploadfile(filename: str) -> str: #input - filename
    command = f"""
    select type, description from ocr where filename = '{filename}';
    """
    cur.execute(command)
    tp, description = cur.fetchone()
    conn.commit()
    conn.close()

    response = model.generate_content(f"""
    Summarise the content of the following key-value pair in 100 words:

    {tp}: {description}           
    """)

    return response.text