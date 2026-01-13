import os
import pickle
import faiss
from chunking import chunk_text  # assuming your chunking.py has this
from embedding import get_embedding

DATA_PATH = "data"
STORE_PATH = "faiss_store"
os.makedirs(STORE_PATH, exist_ok=True)

texts = []
chunk_mapping = []

# Read and chunk all files
for file in os.listdir(DATA_PATH):
    with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
        content = f.read()
        chunks = chunk_text(content, chunk_size=500)  # use your chunking.py function
        texts.extend(chunks)
        chunk_mapping.extend(chunks)

print("Generating embeddings...")
embeddings = [get_embedding(c) for c in texts]
embeddings = np.vstack(embeddings).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(STORE_PATH, "index.faiss"))
with open(os.path.join(STORE_PATH, "chunk_mapping.pkl"), "wb") as f:
    pickle.dump(chunk_mapping, f)

print("FAISS index rebuilt successfully")
