def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        start = end - overlap   # controlled overlap

    return chunks
