import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("🚀 RAG Vector Creation Started...")

documents = []

# Load all TXT files from data folder
for file in os.listdir("data"):
    if file.endswith(".txt"):
        print(f"Loading: {file}")
        loader = TextLoader(
            os.path.join("data", file),
            encoding="utf-8"
        )
        documents.extend(loader.load())

print(f"✅ Total Files Loaded: {len(documents)}")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Total Chunks Created: {len(chunks)}")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding Model Loaded")

# Create FAISS
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_db.save_local("vectorstore")

print("🎉 FAISS Vector Database Created Successfully!")