import streamlit as st
import os
from utils.retrieval import load_faiss_index, retrieve_chunks
from utils.prompt import build_prompt
from utils.completion import generate_completion

# Premium styling and responsive page configurations
st.set_page_config(
    page_title="Knowledge Base RAG Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for glassmorphism and modern UI elements
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Header */
.main-header {
    background: linear-gradient(135deg, #6366F1, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05em;
}

.subtitle {
    color: #8E92B2;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Glassmorphism Answer Box */
.answer-box {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
    margin-top: 10px;
    color: #F3F4F6;
    line-height: 1.6;
    font-size: 1.05rem;
}

/* Source Chunk Cards */
.chunk-card {
    background: rgba(31, 41, 55, 0.5);
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    color: #D1D5DB;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.chunk-title {
    color: #6366F1;
    font-weight: 700;
    margin-bottom: 6px;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# Title section
st.markdown("<div class='main-header'> RAG Model</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A premium retrieval-augmented-generation workspace grounded in Hemant Singh's Knowledge Base</div>", unsafe_allow_html=True)

# Hardcoded to Google Gemini for deployment
provider = "Google Gemini"
model = "gemini-embedding-001"
api_key = os.getenv("GEMINI_API_KEY")

st.sidebar.markdown("### 🛠️ Actions")
force_rebuild = st.sidebar.button(
    "🔄 Rebuild FAISS Index",
    help="Clear local caches and reindex data/hemant.txt using the active embedding model."
)

# Handle Manual Rebuild Trigger
if force_rebuild:
    with st.spinner(f"Re-indexing files with {provider} ({model})..."):
        try:
            index, chunk_mapping = load_faiss_index(
                provider=provider,
                model=model,
                api_key=api_key,
                force_rebuild=True
            )
            st.success(f"✅ FAISS index successfully rebuilt for {provider} using {model}!")
        except Exception as e:
            st.error(f"❌ Failed to rebuild index: {e}")

# Search Input Query
query = st.text_input("💬 Ask a question about the documentation", placeholder="e.g. What is Hemant's background?")

if query:
    try:
        # Load and retrieve relevant chunks
        with st.spinner("Scanning knowledge base..."):
            index, chunk_mapping = load_faiss_index(
                provider=provider,
                model=model,
                api_key=api_key
            )
            top_chunks = retrieve_chunks(
                query,
                index,
                chunk_mapping,
                provider=provider,
                model=model,
                api_key=api_key,
                k=3
            )
        
        if not top_chunks:
            st.warning("⚠️ No matching context found in the uploaded documents.")
        else:
            prompt = build_prompt(top_chunks, query)
            
            # Generate Completion via Groq
            with st.spinner("Synthesizing answer with Groq LLM..."):
                response = generate_completion(prompt)
            
            # Display response
            st.markdown("### 💡 Answer")
            st.markdown(f"<div class='answer-box'>{response}</div>", unsafe_allow_html=True)
            st.write("")
            
            # View Retrieved Chunks
            with st.expander("🔍 View Grounding Chunks"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(
                        f"<div class='chunk-card'><div class='chunk-title'>Source Context {i}</div>{chunk}</div>",
                        unsafe_allow_html=True
                    )
    except Exception as e:
        error_msg = str(e)
        st.error("### ❌ Retrieval or Processing Error")
        st.markdown(f"**Error Details:** `{error_msg}`")
        
        # Proactively explain error and offer solutions
        if "limit" in error_msg.lower() or "403" in error_msg or "forbidden" in error_msg:
            st.info(
                "💡 **Recommended Action:**\n\n"
                "The current daily limit for your Euron API Key has been reached.\n"
                "1. **Switch Provider**: Select **Google Gemini** (pre-configured) or **Hugging Face Inference API** in the left sidebar to bypass this limit.\n"
                "2. **Use a Token**: If you have a Gemini key or Hugging Face API Token, paste it into the credentials field.\n"
                "3. **Re-run**: Click the Rebuild button or re-submit your query."
            )
        elif "api key" in error_msg.lower() or "token" in error_msg.lower() or "401" in error_msg:
            st.info(
                "💡 **Recommended Action:**\n\n"
                "Check the credentials provided in the sidebar. Make sure your Euri API Key or HF Token is correct."
            )
        else:
            st.info(
                "💡 **Recommended Action:**\n\n"
                "1. Try running **Rebuild FAISS Index** in the sidebar to sync the index files.\n"
                "2. Double check your API configurations."
            )
