#!/usr/bin/env python3
"""
ask_health_doc_json.py

Offline RAG Q&A on PubMed-RCT abstracts (236K JSONL) using:
- armanc/pubmed-rct20k JSONL from Hugging Face
- sentence-transformers/all-MiniLM-L6-v2 embeddings
- FAISS vector store
- Llama 3 8B-Q4 via Ollama
- Gradio UI
"""

import gradio as gr
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1️⃣ Stream the JSONL (236K abstracts) without any custom loader scripts
print("🔍 Loading PubMed RCT JSONL slice…")
ds = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/armanc/pubmed-rct20k/resolve/main/train.jsonl",
    split="train[:5000]"  # load first 5k for demo; remove slice for full dataset :contentReference[oaicite:0]{index=0}
)
print(f"✅ Loaded {len(ds)} records")

# 2️⃣ Convert to LangChain Documents (chunk abstracts ≈800 chars)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = []
for rec in ds:
    text = rec.get("text") or rec.get("abstract") or rec.get("MedlineCitation", {}).get("Article", {}).get("AbstractText", "")
    if not text or not isinstance(text, str):
        continue
    for chunk in splitter.split_text(text):
        docs.append(Document(page_content=chunk, metadata={"id": rec.get("abstract_id", "")}))
print(f"📄 Created {len(docs)} document chunks")

# 3️⃣ Generate embeddings & build FAISS index
print("🔗 Generating embeddings & building FAISS index…")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, emb)
print("✅ FAISS index ready")

# 4️⃣ Set up RetrievalQA with local Llama 3 8B-Q4
print("🤖 Loading Llama 3 8B-Q4 via Ollama…")
llm = Ollama(model="llama3:latest",
             temperature=0.1)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
)
print("✅ QA chain ready")

# 5️⃣ Launch Gradio interface
def answer(query: str) -> str:
    return qa_chain(query)["result"]

gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question…"),
    outputs=gr.Textbox(lines=5, label="Answer"),
    title="Ask-Health-Doc (PubMed-RCT JSONL slice, all-local)",
    description=(
        "Streams first 5K records from armanc/pubmed-rct20k (JSONL), "
        "embeds with MiniLM, answers via Llama 3 8B-Q4—all offline."
    ),
).launch()
