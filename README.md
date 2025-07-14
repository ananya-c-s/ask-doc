````markdown
### Ask-Health-Doc — PubMed RAG Q&A (Offline)

Built an **offline RAG pipeline** for medical Q&A on **PubMed abstracts** using:

- **Hugging Face datasets** to stream 5K+ abstracts.
- **MiniLM sentence embeddings** with **FAISS** vector search.
- Local **Llama 3 8B-Q4 via Ollama** for query answering.
- Interactive **Gradio UI** for querying.

Optimized for local, fully self-hosted use—no external API calls

---

## 🛠️ System Requirements

- **Python:** 3.9+
- **RAM:** ≥ 8 GB (16 GB recommended)
- **Disk Space:** ≥ 5 GB (models + embeddings)
- **OS:** Windows, macOS, or Linux
- **Dependencies:** 
  - [Ollama CLI](https://ollama.com)
  - Python libraries (see below)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/ask-health-doc.git
   cd ask-health-doc
````

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required Python packages**

   ```bash
   pip install gradio datasets langchain langchain-community sentence-transformers faiss-cpu
   ```

4. **Install and start Ollama**

   * Follow instructions at [https://ollama.com/download](https://ollama.com/download)
   * Start the Ollama server:

     ```bash
     ollama serve
     ```
   * Pull the Llama 3 model:

     ```bash
     ollama pull llama3:latest
     ```

---

## 🚀 Running the App

```bash
python ask_health_doc_json.py
```

* This streams the first 5000 abstracts from the PubMed-RCT JSONL.
* Chunks abstracts (\~800 chars per chunk) and embeds them.
* Builds a FAISS index for retrieval.
* Answers your medical queries using Llama 3 via Ollama locally.
* Opens a Gradio interface in your browser.

---

## 🎯 Features

* ✅ Purely offline and local pipeline (no external API calls).
* ✅ Quick embeddings with **MiniLM-L6-v2**.
* ✅ Vector search using **FAISS**.
* ✅ Lightweight local inference with **Llama 3 8B-Q4 via Ollama**.
* ✅ User-friendly **Gradio** web interface.

---

## 🧩 Customization

* To load more than 5K abstracts, modify:

  ```python
  split="train[:5000]"
  ```

  Remove or increase the slice for larger datasets, keeping in mind memory constraints.

* To tweak chunking size or overlap:

  ```python
  splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
  ```

* To adjust retrieval depth:

  ```python
  retriever=vectordb.as_retriever(search_kwargs={"k": 3})
  ```

---

## 📚 References

* Dataset: [https://huggingface.co/datasets/armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k)
* Embeddings: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Ollama: [https://ollama.com](https://ollama.com)
* Gradio: [https://gradio.app](https://gradio.app)

---

