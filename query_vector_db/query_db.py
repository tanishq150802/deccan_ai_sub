import google.generativeai as genai
import chromadb
#from db_insertion import Embedfn
import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI\\")

chroma_client = chromadb.PersistentClient() #loading the created vector db
DB_NAME = chroma_client.list_collections()[0]

db = chroma_client.get_collection(name=DB_NAME)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
query = "How did the researchers study the effects of smoke on living tissue?"

result = db.query(query_texts=[query], n_results=1) #gives top most relevent chunk
top_result = result["documents"][0]

print(top_result)

prompt = f"""You are a helpful and informative bot that answers questions using text from 
the reference passage included below. Be sure to respond in a complete sentence, being comprehensive, 
including all relevant background information. Do not pretend to know the answer when it is
not present in the reference passage below.

QUESTION: {query}
PASSAGE: {top_result}
"""

answer = model.generate_content(prompt)
print(answer.text)