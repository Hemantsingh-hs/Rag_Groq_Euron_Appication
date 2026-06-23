import pickle, os, numpy as np, faiss
from utils.embedding import get_embedding
from utils.chunking import chunk_text

def load_faiss_index(provider="Euron", model="text-embedding-3-small", api_key=None, force_rebuild=False):
    safe_model = model.replace('/', '_').replace('\\', '_')
    index_path = f"faiss_store/index_{provider}_{safe_model}.faiss"
    map_path = f"faiss_store/chunk_mapping_{provider}_{safe_model}.pkl"

    # Load existing index if present and rebuild not forced
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(map_path):
        index = faiss.read_index(index_path)
        chunk_mapping = pickle.load(open(map_path, "rb"))
        return index, chunk_mapping

    # Build new index
    os.makedirs("faiss_store", exist_ok=True)

    if not os.path.exists("data/hemant.txt"):
        raise FileNotFoundError("data/hemant.txt not found. Please upload your document.")

    print(f"Building FAISS index for {provider} / {model}...")

    with open("data/hemant.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    vectors = [get_embedding(c, provider=provider, model=model, api_key=api_key) for c in chunks]
    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, index_path)
    pickle.dump(chunks, open(map_path, "wb"))

    return index, chunks

def retrieve_chunks(query, index, chunk_mapping, provider="Euron", model="text-embedding-3-small", api_key=None, k=3):
    qv = get_embedding(query, provider=provider, model=model, api_key=api_key)
    D, I = index.search(np.array([qv]), k)
    
    # Safe retrieval: filter out invalid/negative indices that FAISS might return on empty/small sets
    valid_indices = [i for i in I[0] if 0 <= i < len(chunk_mapping)]
    return [chunk_mapping[i] for i in valid_indices]

