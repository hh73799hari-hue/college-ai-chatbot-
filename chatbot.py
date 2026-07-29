import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load .env
load_dotenv()

print("College AI Chatbot Started...")
print("GROQ KEY:", os.getenv("GROQ_API_KEY"))

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Database
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS Database Loaded Successfully!")

# Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Chat Loop
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Chatbot Closed")
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a College Information Assistant.

Answer the question using ONLY the given context.
If the answer is not available in the context, reply:
"I don't have information about that."

Context:
{context}

Question:
{question}

Give a short and clear answer.
"""

    response = llm.invoke(prompt)

    print("\nBot:", response.content)