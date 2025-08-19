Embeddings Model Mismatch:
The embeddings used to create the FAISS vector store must match the embeddings model used for retrieval. If you upload a PDF when settings.EMBEDDINGS_PROVIDER is "openai", but later search when it is "gemini", the vector store will not work correctly because the embeddings are incompatible.

Empty or Incorrect Vector Store:
If the vector store at settings.VECTOR_DB_PATH was not saved correctly or is empty, retrieval will not return results. Check that process_and_store actually adds documents and saves the vector store.

No Chunks Added:
If the PDF is split into chunks that are already present (due to hashing), no new chunks are added. If the vector store is empty, retrieval will return nothing.

Retriever Parameters Too Restrictive:
The retriever uses "mmr" search with k=5, fetch_k=20, and lambda_mult=0.2. If your vector store has very few documents, these parameters may not return results.

How to Debug:

Check that the vector store file (vector_store) exists and is not empty.
Ensure the same embeddings provider is used for both upload and search.
Print the number of documents in the vector store before retrieval.
Try changing the retriever parameters to be less restrictive (e.g., search_type="similarity", k=1).