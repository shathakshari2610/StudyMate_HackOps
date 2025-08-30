# StudyMate_HackOps
StudyMate is an AI-powered learning assistant designed to help students interact with their study materials more effectively.
📘 StudyMate — PDF Q&A Assistant
🚀 Overview

StudyMate is an AI-powered PDF Q&A assistant.
It allows students and researchers to upload PDFs, build indexes, and ask questions directly from documents.
The app uses Natural Language Processing (NLP) and embeddings to provide accurate answers from PDF contents.

✨ Features

📂 Upload PDFs

⚡ Build searchable indexes

❓ Ask questions directly from the PDF

🔎 Get answers with source references

🎯 Lightweight & user-friendly interface (Gradio)

🛠️ Tech Stack

Python 3.9+

Gradio – Web UI

LangChain – Document Q&A

FAISS / Chroma – Vector storage

OpenAI / HuggingFace embeddings – Semantic understanding

📂 Project Structure
studymatelast/
│── app.py              # Main application
│── requirements.txt    # Dependencies
│── .env                # API keys (not uploaded to GitHub)
│── .gitignore          # Ignore venv & secrets

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/StudyMate_HackOps.git
cd StudyMate_HackOps


Create virtual environment

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Add API Key

Create a .env file and add your key:

OPENAI_API_KEY=your_api_key_here


Run the app

python app.py


Open in browser → http://127.0.0.1:7860

📸 Screenshots

(Add screenshots of your app here for better visibility!)

🔮 Future Improvements

Multi-PDF Q&A support

Export summarized notes

Cloud-based storage

Integration with Google Drive / OneDrive
