from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_folder = Path("data")

all_text = ""

# Read all text files
for file in data_folder.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        all_text += f.read() + "\n"

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(all_text)

print(f"Total Chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}")
    print("-" * 50)
    print(chunk)
    print()