<<<<<<< HEAD
import requests, os, numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def get_embedding(text, provider="Euron", model="text-embedding-3-small", api_key=None):
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    
    if provider == "Euron":
        key = api_key or os.getenv("EURI_API_KEY")
        if not key:
            try:
                key = st.secrets["EURI_API_KEY"]
            except Exception:
                pass
        if not key:
            raise ValueError("Euri API Key is missing. Please enter it in the sidebar or set EURI_API_KEY in your environment.")

        r = requests.post(
            "https://api.euron.one/api/v1/euri/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "input": text},
            timeout=60
        )
        
        try:
            data = r.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from Euron: {e}, response text: {r.text}")
        
        if "data" not in data:
            raise ValueError(f"Euron API returned error or unexpected format: {data}")
        
        return np.array(data['data'][0]['embedding'], dtype="float32")

    elif provider == "Hugging Face Inference API":
        key = api_key or os.getenv("HF_TOKEN")
        if not key:
            try:
                key = st.secrets["HF_TOKEN"]
            except Exception:
                pass
        
        # Use router.huggingface.co endpoint which resolves reliably
        api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        
        r = requests.post(
            api_url,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=60
        )
        
        try:
            data = r.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from Hugging Face: {e}, response text: {r.text}")
        
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Hugging Face API returned error: {data['error']}")
        
        # Recursive extraction of the first list of floats (handles various nests like [[float]] or [[[float]]])
        def extract_vector(obj):
            if isinstance(obj, list):
                if len(obj) == 0:
                    return None
                if isinstance(obj[0], (int, float)):
                    return obj
                return extract_vector(obj[0])
            return None
        
        vector = extract_vector(data)
        if vector is None:
            raise ValueError(f"Hugging Face API response did not contain a valid embedding vector: {data}")
        
        return np.array(vector, dtype="float32")

    elif provider == "Google Gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            try:
                key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass
        if not key:
            raise ValueError("Google Gemini API Key is missing. Please enter it in the sidebar or set GEMINI_API_KEY in your environment.")

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}"
        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        r = requests.post(api_url, json=payload, timeout=60)
        try:
            data = r.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from Gemini: {e}, response text: {r.text}")
        
        if "error" in data:
            raise ValueError(f"Gemini API returned error: {data['error'].get('message', data['error'])}")
        
        if "embedding" not in data or "values" not in data["embedding"]:
            raise ValueError(f"Gemini API returned unexpected format: {data}")
        
        return np.array(data["embedding"]["values"], dtype="float32")

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

=======
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):

    if not text.strip():
        raise ValueError("Empty text")

    embedding = model.encode(text)

    return np.array(embedding, dtype="float32")


# import requests
# import os
# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("EURI_API_KEY")

# if not API_KEY:
#     try:
#         API_KEY = st.secrets["EURI_API_KEY"]
#     except Exception:
#         API_KEY = None


# def get_embedding(text, model="text-embedding-3-small"):

#     if not text.strip():
#         raise ValueError("Cannot embed empty text")

#     if not API_KEY:
#         raise ValueError("Missing EURI_API_KEY")

#     response = requests.post(
#         "https://api.euron.one/api/v1/euri/embeddings",
#         headers={
#             "Authorization": f"Bearer {API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": model,
#             "input": text
#         },
#         timeout=60
#     )

#     try:
#         data = response.json()
#     except Exception as e:
#         raise ValueError(
#             f"Failed to parse JSON: {e}, response text: {response.text}"
#         )

#     # Handle API errors
#     if response.status_code != 200:

#         error_message = data.get("error", {}).get(
#             "message",
#             "Unknown API error"
#         )

#         raise ValueError(
#             f"Euron API Error ({response.status_code}): {error_message}"
#         )

#     # Validate response format
#     if "data" not in data:
#         raise ValueError(
#             f"Unexpected API response format: {data}"
#         )

#     embedding = data["data"][0]["embedding"]

#     return np.array(embedding, dtype="float32")


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

# import requests, os, numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# try:
#     API_KEY = os.getenv("EURI_API_KEY")
#     if not API_KEY:
#         API_KEY = st.secrets["EURI_API_KEY"]
# except Exception:
#     pass

# def get_embedding(text, model="text-embedding-3-small"):
#     if not text.strip():
#         raise ValueError("Cannot embed empty text")
    
#     r = requests.post(
#         "https://api.euron.one/api/v1/euri/embeddings",
#         headers={"Authorization": f"Bearer {API_KEY}"},
#         json={"model": model, "input": text},
#         timeout=60
#     )
    
#     try:
#         data = r.json()
#     except Exception as e:
#         raise ValueError(f"Failed to parse JSON: {e}, response text: {r.text}")
    
#     if "data" not in data:
#         raise ValueError(f"Euron API returned error or unexpected format: {data}")
    
    # return np.array(data['data'][0]['embedding'], dtype="float32")
>>>>>>> b0b9ec9348e2f4c41330bb512e4206a61c00945e
