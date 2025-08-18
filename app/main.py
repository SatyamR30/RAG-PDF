from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import shutil
import os
from app.embeddings import process_and_store
from app.rag_pipeline import query_rag
from app.database import init_db, insert_metadata, get_metadata

app = FastAPI(title="RAG Assignment")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup():
    init_db()

@app.post("/upload")
async def upload_docs(files: List[UploadFile]):
    for file in files:
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Process and store embeddings
        process_and_store(filepath)
        insert_metadata(file.filename, filepath)
    return {"status": "success", "uploaded_files": [f.filename for f in files]}

@app.post("/query")
async def query_system(query: str = Form(...)):
    answer = query_rag(query)
    return {"query": query, "answer": answer}

@app.get("/metadata")
def get_docs_metadata():
    return get_metadata()
