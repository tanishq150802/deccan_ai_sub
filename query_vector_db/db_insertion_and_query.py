import google.generativeai as genai
import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from chromadb.config import Settings
import PIL.Image
from dotenv import load_dotenv
from classes import Embedfn
# from class import ImagePreprocessor

import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")

to_be_vectorized = "dataset\\Scientific\\01142172_01142173.jpg"
file = PIL.Image.open(to_be_vectorized)

##### Using OpenCV Image processing + Pytesseract for OCR
# process = ImagePreprocessor(to_be_vectorized) 
# doc = process.preprocess_and_extract()
# documents = doc.split(".")

##### Using MLLM for OCR
model = genai.GenerativeModel(model_name="gemini-1.5-flash") 
prompt = "Extract the text shown in this image."
response = model.generate_content([prompt, file])
documents = response.text.split("\n\n") #chunking

DB_NAME = "rag"
embed_fn = Embedfn()
embed_fn.document_mode = False #testing if the question gives relevent chunks based on query

chroma_client = chromadb.PersistentClient(path="chromadb_store") #saving to disk
# chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
#                                      persist_directory=os.getcwd()
#                                 ))
db = chroma_client.get_or_create_collection(name=DB_NAME, 
                                    embedding_function=embed_fn)

db.add(documents=documents, ids=[str(i) for i in range(len(documents))]) #inserting the vectorized chunks

query = "How did the researchers study the effects of smoke on living tissue?"

result = db.query(query_texts=[query], n_results=2) #gives top 2 most relevent chunks
top_results = result["documents"][0]

# print(top_results)

#['Inspired by this similarity, the researchers decided to use ciliated tissues from the 
# mouth and esophagus of the frog in their effort to measure certain effects of smoke exposure 
# on living tissues. They observed that specks of carbon placed at one end of a specimen of 
# frog tissue are carried to the opposite end on mucus currents propelled by cilia.', 
# 'Chemists tested a number of experimental compounds designed to "capture" the phenol from 
# smoke particles. One particular additive incorporated into a filter proved to be 
# strikingly effective in trapping the phenol. Frog tissue']

### Using RAG for querying
rag_query = "Describe the correlation between smoke and cilia depression."
result = db.query(query_texts=[rag_query], n_results=1) #gives top most relevent chunk

passage = result['documents'][0][0]

prompt = f"""You are a helpful and informative bot that answers questions using text from 
the reference passage included below. Be sure to respond in a complete sentence, being comprehensive, 
including all relevant background information. Do not pretend to know the answer when it is
not present in the reference passage below.

QUESTION: {rag_query}
PASSAGE: {passage}
"""

answer = model.generate_content(prompt)
print(answer.text)
# Studies have shown a direct correlation between the phenol content in smoke and the 
# depression of ciliary beating and the slowing of mucus flow;  higher phenol levels 
# in smoke correlate with greater depression of cilia.  This suggests that phenol is the primary, 
# or perhaps sole, component of smoke responsible for this effect.