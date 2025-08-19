from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from app.config import settings

def get_embeddings_model():
    if settings.EMBEDDINGS_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
    return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

def get_llm():
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)
    if settings.LLM_PROVIDER == "groq":
        return ChatGroq(model_name="llama-3.1-70b-versatile", groq_api_key=settings.GROQ_API_KEY, temperature=0.2)
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=settings.OPENAI_API_KEY)

def query_rag(query):
    embedding_model = get_embeddings_model()
    db = FAISS.load_local(settings.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.2}
    )
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
