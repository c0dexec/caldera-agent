from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveJsonSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import requests

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# -------- Load Swagger JSON --------
json_data = requests.get(
    "http://12.1.0.15:8888/api/docs/swagger.json"
).json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_docs = json_splitter.create_documents([json_data])

# -------- Load Markdown --------
def md_data():
    with open("response.md", "r") as f:
        return f.read()

markdown_splitter = MarkdownHeaderTextSplitter(
    [("###", "Header 3")],
    strip_headers=False
)

markdown_docs = markdown_splitter.split_text(md_data())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30
)

md_docs = text_splitter.split_documents(markdown_docs)

# -------- Build FAISS index (THIS creates the index) --------
all_docs = json_docs + md_docs

vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

vector_store.save_local("faiss_caldera_vectorstore")