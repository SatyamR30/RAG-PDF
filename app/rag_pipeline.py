from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from app.config import settings

def get_embeddings_model():
    if settings.LLM_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
    return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

def get_llm():
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=settings.OPENAI_API_KEY)

def query_rag(query):
    embedding_model = get_embeddings_model()
    db = FAISS.load_local(settings.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 15})
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
