from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from app.config import settings
from app.rag_pipeline import get_embeddings_model
import os

def load_document(filepath):
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        toc_docs = docs[:10]
        return docs + toc_docs
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
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(settings.VECTOR_DB_PATH)
