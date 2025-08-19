from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.config import settings
from app.rag_pipeline import get_embeddings_model
import os
import hashlib

def load_document(filepath):
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        return docs
    else:
        loader = TextLoader(filepath)
        return loader.load()


def process_and_store(filepath):
    documents = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embedding_model = get_embeddings_model()

    if os.path.exists(settings.VECTOR_DB_PATH):
        db = FAISS.load_local(settings.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        try:
            existing_docs = list(db.docstore._dict.values())
            existing_hashes = {
                hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
                for doc in existing_docs
            }
        except Exception:
            existing_hashes = set()

        new_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
            if chunk_hash not in existing_hashes:
                new_chunks.append(chunk)

        if new_chunks:
            db.add_documents(new_chunks)
    else:
        db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(settings.VECTOR_DB_PATH)
