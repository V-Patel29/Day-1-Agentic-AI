import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import requests

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

# Sample documents
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for AI.",
    "Groq is a platform that runs LLMs at blazing speed.",
    "The capital of Japan is Tokyo."
]

# Initialize sentence transformer model for embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Build FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Query
query = "Where is the Eiffel Tower?"

# Embed query
query_embedding = embedder.encode([query], convert_to_numpy=True)

# Search
k = 2
distances, indices = index.search(query_embedding, k)
retrieved_chunks = [documents[i] for i in indices[0]]

# Combine context
context = "\n".join(retrieved_chunks)
prompt = f"""Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

# Call Groq API
def query_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Run it
answer = query_groq(prompt)
print("Answer:", answer)
