from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
def load_docs():
    with open("data/docs.txt", "r") as f:
        return f.read().split("\n\n")

docs = load_docs()

# Create embeddings
doc_embeddings = model.encode(docs)

# Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Load local LLM
generator = pipeline("text-generation", model="distilgpt2")

# LLM function
def generate_answer(query, context):
    prompt = f"""
You are an expert debugging assistant.

User query: {query}

Relevant logs:
{context}

Give a clear explanation and solution:
"""

    result = generator(
        prompt,
        max_length=150,
        do_sample=True,
        temperature=0.7
    )

    return result[0]['generated_text']


# Query function
def query_rag(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = [docs[i] for i in indices[0]]
    context = "\n".join(results)

    answer = generate_answer(query, context)
    return answer