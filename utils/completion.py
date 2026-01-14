from groq import Groq
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_completion(prompt, model="llama-3.1-8b-instant", temperature=0.1):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer strictly from context."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
