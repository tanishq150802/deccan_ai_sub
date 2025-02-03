import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings
from vectordb_insertion import embedding_fn
warnings.filterwarnings('ignore')
os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")
from pymilvus import MilvusClient

load_dotenv()

question = "Which compound is a major cilia-depressing agent in smoke?"

milvus_client = MilvusClient(uri="milvus_db.db")
search_grid = milvus_client.search(
    collection_name="rag",
    data=[
        embedding_fn(question)
    ],  
    limit = 3,  
    search_params={"metric_type": "COSINE", "params": {}},
    output_fields=["text"]
)

retrieved_lines_with_similarity = [
    (res["entity"]["text"], res["distance"]) for res in search_grid[0]
]

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_similarity]
)

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

model = genai_model()
response = model.generate_content(f"""
You are given a context and question below. You need to answer the question based on the next only.

"Question": {question}
"Context": {context}           
""")

print(response.text)