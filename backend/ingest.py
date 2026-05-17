import os
import pickle
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.pkl"
CHUNK_SIZE = 800      # was 500
CHUNK_OVERLAP = 100   # was 50

def get_embedding(text: str) -> list:
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

def build_index(pdf_folder: str = "data/pdfs"):
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            text = parse_pdf(os.path.join(pdf_folder, filename))
            all_chunks.extend([(filename, c) for c in chunk_text(text)])
    if not all_chunks:
        print("No PDFs found.")
        return
    texts = [c for _, c in all_chunks]
    embeddings = np.array([get_embedding(t) for t in texts]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs("data/index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Done! {len(all_chunks)} chunks indexed.")

if __name__ == "__main__":
    build_index()