from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveJsonSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import requests
import glob

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
markdown_files = glob.glob("./docs/*.md")  # matches all Markdown files in current folder
markdown_texts = []

for file_path in markdown_files:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_texts.append(f.read())

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)

markdown_docs = []
for text in markdown_texts:
    split_docs = markdown_splitter.split_text(text)
    split_docs[0]
    markdown_docs.extend(split_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30
)

md_docs = text_splitter.split_documents(markdown_docs)

all_docs = json_docs + md_docs

vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

vector_store.save_local("faiss_caldera_vectorstore")