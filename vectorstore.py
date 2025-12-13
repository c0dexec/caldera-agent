from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

_embeddings = None
_vector_store = None

def get_vector_store():
    global _vector_store, _embeddings

    if _vector_store is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        _vector_store = FAISS.load_local(
            "faiss_caldera_vectorstore",
            _embeddings,
            allow_dangerous_deserialization=True
        )

    return _vector_store
