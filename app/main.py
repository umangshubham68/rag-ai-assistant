from fastapi import FastAPI
from app.rag import query_rag

app = FastAPI()

@app.get("/")
def home():
  return {"message": "Welcome to the RAG API!"}

@app.get("/query")
def query(q: str):
  results = query_rag(q)
  return {"answer": results}