import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# Find data folder
# --------------------------------

DATA_FOLDER = "data"

if not os.path.exists(DATA_FOLDER):
    print("❌ data folder not found!")
    exit()

print("📚 Reading college data...")


# --------------------------------
# Read all TXT files
# --------------------------------

all_text = ""

for filename in os.listdir(DATA_FOLDER):

    if filename.endswith(".txt"):

        filepath = os.path.join(DATA_FOLDER, filename)

        print(f"Reading: {filename}")

        with open(filepath, "r", encoding="utf-8") as file:
            all_text += "\n" + file.read()


# --------------------------------
# Split text into chunks
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.split_text(all_text)

print(f"Total chunks created: {len(documents)}")


# --------------------------------
# Create TF-IDF Vectorizer
# --------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

vectors = vectorizer.fit_transform(documents)


# --------------------------------
# Save lightweight vector database
# --------------------------------

os.makedirs("vectorstore", exist_ok=True)

with open("vectorstore/data.pkl", "wb") as file:

    pickle.dump(
        {
            "documents": documents,
            "vectorizer": vectorizer,
            "vectors": vectors
        },
        file
    )


print("✅ Lightweight vector database created!")
print("📁 Saved inside: vectorstore/data.pkl")