import faiss, pickle, os, numpy as np
from utils.embedding import get_embedding
from utils.chunking import chunk_text

INDEX_PATH = "faiss_store/index.faiss"
MAP_PATH = "faiss_store/chunk_mapping.pkl"

def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH):
        index = faiss.read_index(INDEX_PATH)
        chunk_mapping = pickle.load(open(MAP_PATH, "rb"))
        return index, chunk_mapping

    print("Building FAISS index...")
    text = open("data/hemant.txt", "r", encoding="utf-8").read()
    chunks = chunk_text(text)

    vectors = [get_embedding(c) for c in chunks]
    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    pickle.dump(chunks, open(MAP_PATH, "wb"))

    return index, chunks

def retrieve_chunks(query, index, chunk_mapping, k=3):
    qv = get_embedding(query)
    D, I = index.search(np.array([qv]), k)
    return [chunk_mapping[i] for i in I[0]]
