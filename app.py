import os
import pickle

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq


# --------------------------------
# Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# Flask App
# --------------------------------

app = Flask(__name__)


# --------------------------------
# Load Lightweight Vector Database
# --------------------------------

print("Loading lightweight vector database...")


with open("vectorstore/data.pkl", "rb") as file:

    vector_data = pickle.load(file)


documents = vector_data["documents"]
vectorizer = vector_data["vectorizer"]
vectors = vector_data["vectors"]


print("✅ Vector database loaded.")
print(f"Total chunks: {len(documents)}")


# --------------------------------
# Load Groq LLM
# --------------------------------

print("Loading Groq LLM...")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)


print("✅ Groq LLM loaded.")


# --------------------------------
# Home Page
# --------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------
# Chat API
# --------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        question = data.get("message", "").strip()


        if not question:

            return jsonify({
                "answer": "Please enter a question."
            })


        # --------------------------------
        # Convert question to TF-IDF
        # --------------------------------

        question_vector = vectorizer.transform(
            [question]
        )


        # --------------------------------
        # Calculate similarity
        # --------------------------------

        similarities = cosine_similarity(
            question_vector,
            vectors
        )[0]


        # --------------------------------
        # Get top 3 relevant chunks
        # --------------------------------

        top_indices = similarities.argsort()[-3:][::-1]


        selected_documents = []

        for index in top_indices:

            # Ignore very weak matches

            if similarities[index] > 0:

                selected_documents.append(
                    documents[index]
                )


        # --------------------------------
        # No relevant information
        # --------------------------------

        if not selected_documents:

            return jsonify({
                "answer": "I don't have information about that."
            })


        # --------------------------------
        # Build context
        # --------------------------------

        context = "\n\n".join(
            selected_documents
        )


        # --------------------------------
        # RAG Prompt
        # --------------------------------

        prompt = f"""
You are an AI College Information Chatbot.

Answer the user's question ONLY using the CONTEXT provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent information.
4. If the answer is not explicitly available in the context, reply exactly:
"I don't have information about that."
5. Give short and clear answers.
6. If multiple items are available, list them clearly.
7. For fees, provide only the fee mentioned in the context.
8. Do not add facilities, courses, fees or details that are not present in the context.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


        # --------------------------------
        # Generate answer
        # --------------------------------

        response = llm.invoke(prompt)


        return jsonify({
            "answer": response.content
        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "answer": "Sorry, something went wrong."
        })


# --------------------------------
# Run Flask
# --------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )