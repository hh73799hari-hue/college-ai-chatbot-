from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()

# Read all text files
data_folder = Path("data")
all_text = ""

for file in data_folder.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        all_text += f.read() + "\n"

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(all_text)

print(f"Total Chunks: {len(chunks)}")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vector_db = FAISS.from_texts(chunks, embeddings)

# Save database
vector_db.save_local("faiss_index")

print("✅ FAISS Vector Database Created Successfully!")