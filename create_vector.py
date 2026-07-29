from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


print("RAG Vector Creation Started...")


# Load Text File
loader = TextLoader("data/About_College.txt", encoding="utf-8")
documents = loader.load()

print("Text File Loaded Successfully")
print("Total Documents:", len(documents))


# Split Text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Text Chunks Created:", len(chunks))


# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


# Create FAISS Vector Database
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)


# Save Database
vector_db.save_local("vectorstore")

print("FAISS Vector Database Created Successfully!")