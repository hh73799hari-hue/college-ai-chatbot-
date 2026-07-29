import os

# Prevent OpenBLAS memory issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load .env variables
load_dotenv()

app = Flask(__name__)

# Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Database
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS Database Loaded Successfully!")

# Load Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    question = data.get("message", "")

    # Search relevant documents
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a College Information Assistant.

Answer ONLY using the given context.

If the answer is not available in the context, reply:
"I don't have information about that."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return jsonify({
        "answer": response.content
    })


# Run Flask App
if __name__ == "__main__":
    print("🚀 College AI Chatbot Started...")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )