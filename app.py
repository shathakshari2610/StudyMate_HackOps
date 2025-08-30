import os, json, pickle, textwrap, pathlib
import numpy as np
import fitz  # PyMuPDF
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from huggingface_hub import login

# ---------------------------
# HuggingFace login
# ---------------------------
# Reuse token already saved on your system (from hf login)
login(token=None)

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # updated to full name
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ---------------------------
# PDF utils
# ---------------------------
def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(texts)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if end == len(text): break
    return chunks

# ---------------------------
# Embeddings + FAISS
# ---------------------------
# ✅ Fixed: embedder uses auth token
_embedder = SentenceTransformer(EMBED_MODEL_NAME, use_auth_token=True)

def embed_texts(chunks):
    embs = _embedder.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embs.astype("float32")

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

def search(index, query, chunks, top_k=TOP_K):
    q_emb = _embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    return [(chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]

# ---------------------------
# LLM calls (HuggingFace)
# ---------------------------
def call_huggingface_inference(prompt, max_new_tokens=256, temperature=0.2, timeout=60):
    if not HF_TOKEN:
        return "⚠️ Set HF_TOKEN in .env first."
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": f"[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
    try:
        r = requests.post(HF_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"(HF error) {e}"

def build_prompt(question, contexts):
    joined = "\n\n---\n\n".join(contexts)
    prompt = f"""You are StudyMate. Answer ONLY using context below. If not in context, say you don't know.

Context:
{joined}

Question: {question}
Answer:"""
    return prompt

# ---------------------------
# Save/Load index
# ---------------------------
def save_index(index, chunks, path="studymate.index"):
    faiss.write_index(index, path + ".faiss")
    with open(path + ".meta.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "model": EMBED_MODEL_NAME}, f)

def load_index(path="studymate.index"):
    idx = faiss.read_index(path + ".faiss")
    with open(path + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return idx, meta["chunks"]

# ---------------------------
# Gradio UI
# ---------------------------
def ui_build_index(files, use_folder):
    chunks, filepaths = [], []
    if use_folder:
        filepaths += [str(p) for p in pathlib.Path("data").glob("*.pdf")]
    if files:
        filepaths += [f.name for f in files]
    if not filepaths:
        return None, None, "❌ No PDFs."
    texts = [extract_text_from_pdf(p) for p in filepaths]
    for t in texts:
        chunks += chunk_text(t)
    embs = embed_texts(chunks)
    index = build_faiss_index(embs)
    return index, chunks, f"✅ Indexed {len(filepaths)} file(s)."

def ui_ask(question, chatbot, index, chunks, show_sources):
    if index is None:
        return chatbot + [["You", question], ["StudyMate", "⚠️ Build index first."]]
    results = search(index, question, chunks, TOP_K)
    contexts = [r[0] for r in results]
    prompt = build_prompt(question, contexts)
    answer = call_huggingface_inference(prompt)
    if show_sources:
        answer += "\n\nSources:\n" + "\n".join([f"- {c[:150]}..." for c in contexts])
    return chatbot + [["You", question], ["StudyMate", answer.strip()]]

with gr.Blocks(title="StudyMate") as demo:
    gr.Markdown("# 📘 StudyMate — PDF Q&A")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
            use_folder = gr.Checkbox(label="Use PDFs from ./data", value=True)
            build_btn = gr.Button("Build Index")
            status = gr.Markdown()
            save_btn = gr.Button("Save Index")
            load_btn = gr.Button("Load Index")
            show_sources = gr.Checkbox(label="Show sources", value=True)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            question = gr.Textbox(label="Ask a question")
            ask_btn = gr.Button("Ask")

    index_state, chunks_state = gr.State(None), gr.State(None)
    build_btn.click(fn=ui_build_index, inputs=[files, use_folder], outputs=[index_state, chunks_state, status])
    ask_btn.click(fn=ui_ask, inputs=[question, chatbot, index_state, chunks_state, show_sources], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()


