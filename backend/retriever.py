import pickle
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.pkl"

def get_embedding(text: str) -> list:
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def retrieve(query: str, top_k: int = 4) -> list[dict]:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    query_vec = np.array([get_embedding(query)]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [{"source": chunks[i][0], "text": chunks[i][1]} for i in indices[0] if i < len(chunks)]