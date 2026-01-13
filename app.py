import streamlit as st
from utils.retrieval import load_faiss_index, retrieve_chunks
from utils.prompt import build_prompt
from utils.completion import generate_completion

st.set_page_config(page_title="Hybrid RAG", layout="wide")

st.title("Hybrid RAG – Hemant Singh Knowledge Base")
st.write("Ask questions grounded only in the uploaded documents.")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching documents..."):
        index, chunk_mapping = load_faiss_index()
        top_chunks = retrieve_chunks(query, index, chunk_mapping, k=3)

    prompt = build_prompt(top_chunks, query)

    with st.spinner("Generating answer..."):
        response = generate_completion(prompt)

    st.subheader("Answer")
    st.write(response)

    with st.expander("View Retrieved Chunks"):
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
