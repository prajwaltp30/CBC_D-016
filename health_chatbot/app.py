from flask import Flask, request, jsonify, render_template, session
import fitz  # PyMuPDF
import requests
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def ask_groq(prompt):
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful health report assistant.Summarize the following health report in simple, short bullet points (max 50 words per point).Avoid complicated medical terms if possible.Make it very easy to understand for a normal person.Report content:"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error contacting Groq AI: {response.text}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    extracted_text = extract_text_from_pdf(file)
    session['report_text'] = extracted_text  # Save report text for chat
    ai_response = ask_groq(f"Summarize and explain this health report:\n{extracted_text}")
    
    return jsonify({"ai_response": ai_response})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    report_text = session.get('report_text')

    if not report_text:
        return jsonify({"error": "No report uploaded yet."}), 400

    full_prompt = f"Given this health report:\n{report_text}\n\nUser's Question: {user_message}\nYou are a health report AI assistant.Based on the uploaded report, answer the following user question shortly (less than 50 words), clearly, and simply:Question: {user_message}"
    ai_response = ask_groq(full_prompt)
    
    return jsonify({"ai_response": ai_response})

if __name__ == "__main__":
    app.run(debug=True, port = 5001)
