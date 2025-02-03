from preprocessing import ImagePreprocessor
import os, json
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

os.chdir("C:\\Old PC\\Mission Machine Learning\\Deccan_AI")
milvus_client = MilvusClient(uri="milvus_db.db")

collection_name = "rag"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=768,
    metric_type="COSINE",  
    consistency_level="Strong" 
)

to_be_vectorized = "dataset\\Scientific\\01142172_01142173.jpg"

process = ImagePreprocessor(to_be_vectorized)
doc = process.preprocess_and_extract()
lines = doc.split(".")

def embedding_fn(text):
    result = genai.embed_content(
            model="models/text-embedding-004",
            content=text)
    return result["embedding"]

data = []
for i in tqdm(range(len(lines))):
    data.append({"id": i, "vector": embedding_fn(lines[i]), "text": lines[i]})

milvus_client.insert(collection_name=collection_name, data=data)

## test 
question = "Which compound is a major cilia-depressing agent in smoke?"

search_grid = milvus_client.search(
    collection_name=collection_name,
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
print(json.dumps(retrieved_lines_with_similarity, indent=4))