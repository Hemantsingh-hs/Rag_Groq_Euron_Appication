# import requests, os, numpy as np
# from dotenv import load_dotenv
# load_dotenv()

# API_KEY = os.getenv("EURI_API_KEY")

# def get_embedding(text, model="text-embedding-3-small"):
#     r = requests.post(
#         "https://api.euron.one/api/v1/euri/embeddings",
#         headers={"Authorization": f"Bearer {API_KEY}"},
#         json={"model": model, "input": text},
#         timeout=60
#     )
#     data = r.json()
#     return np.array(data['data'][0]['embedding'], dtype="float32")

import requests, os, numpy as np
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("EURI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    
    r = requests.post(
        "https://api.euron.one/api/v1/euri/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "input": text},
        timeout=60
    )
    
    try:
        data = r.json()
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}, response text: {r.text}")
    
    if "data" not in data:
        raise ValueError(f"Euron API returned error or unexpected format: {data}")
    
    return np.array(data['data'][0]['embedding'], dtype="float32")
