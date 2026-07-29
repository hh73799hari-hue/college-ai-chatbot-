import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)

# College Information
college_context = """
SHANMUGA INDUSTRIES ARTS AND SCIENCE COLLEGE

About:
Shanmuga Industries Arts and Science College (SIASC) is located in Tiruvannamalai District, Tamil Nadu.

Established:
1996

Affiliation:
Thiruvalluvar University

Courses Offered:
1. B.Sc Data Science
2. B.Sc Computer Science
3. BCA
4. B.Com
5. BBA
6. BA English
7. BA Tamil

Facilities:
Library
Computer Lab
Hostel
Transport
Placement Cell
Sports
Wi-Fi Campus
"""

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data["message"]

    prompt = f"""
You are an AI College Information Chatbot.

Answer ONLY using the information below.

If the answer is not available, reply:

"I don't have information about that."

College Information:

{college_context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return jsonify({"answer": response.content})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)