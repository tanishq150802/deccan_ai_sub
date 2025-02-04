import chromadb
import google.generativeai as genai
import chainlit as cl
from classes import Embedfn
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash") 

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")

chroma_client = chromadb.PersistentClient(path="chromadb_store") #reloading from disk

def ask(query: str) -> str:
    db = chroma_client.get_or_create_collection(name="rag", 
                                                embedding_function=Embedfn())
    result = db.query(query_texts=[query], n_results=3) #gives top 3 most relevent chunks
    passage = ""

    for p in result['documents'][0]: #concatanating chunks to produce context
        passage = passage + '\n' + p
    prompt = f"""You are a helpful and informative bot that answers questions using text from 
    the reference passage included below. Be sure to respond in a complete sentence, being comprehensive, 
    including all relevant background information. Do not pretend to know the answer when it is
    not present in the reference passage below.

    QUESTION: {query}
    PASSAGE: {passage}
    """
    answer = model.generate_content(prompt)
    return answer.text

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="How are smoke and cilia depression related?",
            message="Describe the correlation between smoke and cilia depression.",
            icon = "query_vector_db\\frog-svgrepo-com.svg"
            )
        ]

@cl.on_message
async def on_message(message: cl.Message):
    response = ask(message.content)
    await cl.Message(response).send()