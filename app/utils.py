from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into manageable chunks for embedding.

    Args:
        documents (List[Document]): LangChain Document objects.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
