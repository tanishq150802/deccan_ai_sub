import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings
import sqlite3
warnings.filterwarnings('ignore')
os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")
conn = sqlite3.connect("sql_db.db")
cur = conn.cursor()

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def genai_model(model_str: str = 'gemini-1.5-flash', temp: int = 1, top_p: float = 0.95, 
                top_k: int = 40):
    """Initializes model."""
    model = genai.GenerativeModel(
        model_str,
        generation_config=genai.GenerationConfig(
            temperature = temp,
            top_k = top_k,
            top_p = top_p
        ))
    return model

filename = "2064207315c.jpg" 
command = f"""
select type, description from ocr where filename = '{filename}';
"""
cur.execute(command)
tp, description = cur.fetchone()
conn.commit()
conn.close()

model = genai_model()
response = model.generate_content(f"""
Summarise the content of the following key-value pair in 100 words:

{tp}: {description}           
""")

print(response.text)