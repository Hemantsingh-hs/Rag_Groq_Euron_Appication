def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""You must answer ONLY from the context below.
If the answer is not in the context, say: "Not found in document."

Context:
{context}

Question:
{query}

Answer:"""
