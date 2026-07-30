import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = Flask(__name__)

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded.")

# -----------------------------
# Load FAISS Vector Database
# -----------------------------
print("Loading FAISS Vector Database...")

vector_db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

print("FAISS Vector Database Loaded.")

# -----------------------------
# Load Groq LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Chat API
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    question = data.get("message", "")

    try:

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an AI College Information Chatbot.

Answer ONLY using the information given below.

Rules:

1. Give short and accurate answers.
2. If the answer is unavailable, reply:
"I don't have information about that."
3. Do not create your own answers.
4. If fees are asked, provide the exact fee.
5. If multiple courses are asked, list them clearly.

College Information:

{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        return jsonify({
            "answer": response.content
        })

    except Exception as e:

        return jsonify({
            "answer": str(e)
        })

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)