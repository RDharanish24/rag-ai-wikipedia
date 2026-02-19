Wikipedia RAG (Local LLM + Streamlit)

A simple Retrieval-Augmented Generation (RAG) application built with:

🧠 LlamaIndex

🤗 HuggingFace Embeddings

🔥 Qwen 2.5 (0.5B Instruct) Local LLM

📚 Wikipedia as Knowledge Source

🌐 Streamlit UI

This app allows you to ask questions about AI / Machine Learning topics, and it answers using Wikipedia content with a local LLM.

🚀 Features

✅ Fully Local LLM (No OpenAI API required)

✅ Wikipedia document retrieval

✅ HuggingFace embeddings (all-MiniLM-L6-v2)

✅ Persistent vector index storage

✅ Streamlit web interface

✅ CPU compatible (slower but works)

📚 Topics Included

The RAG index is built from the following Wikipedia pages:

Artificial intelligence

Machine learning

Deep learning

Convolutional neural network

Long short-term memory

🛠️ Tech Stack
Component	Tool Used
Framework	Streamlit
RAG Engine	LlamaIndex
Embeddings	sentence-transformers/all-MiniLM-L6-v2
LLM	Qwen/Qwen2.5-0.5B-Instruct
Data Source	Wikipedia
📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/wiki-rag.git
cd wiki-rag

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Requirements
pip install -r requirements.txt


If you don't have a requirements.txt, install manually:

pip install streamlit torch llama-index llama-index-embeddings-huggingface \
llama-index-llms-huggingface llama-index-readers-wikipedia \
sentence-transformers transformers

▶️ Run the App
streamlit run app.py


Then open:

http://localhost:8501

📁 Project Structure
wiki-rag/
│
├── app.py
├── wiki_rag/          # Persisted vector index
├── README.md
└── requirements.txt

⚙️ How It Works

Wikipedia pages are loaded using WikipediaReader

Documents are chunked (512 tokens)

Text embeddings are created using all-MiniLM-L6-v2

Vector index is stored locally (./wiki_rag)

When a question is asked:

Top similar chunk is retrieved

Qwen LLM generates answer from context

Retrieved context is displayed in UI

🧠 Model Details
Embedding Model

sentence-transformers/all-MiniLM-L6-v2

Lightweight and fast

Good for CPU usage

LLM

Qwen/Qwen2.5-0.5B-Instruct

Decoder-only model

2048 context window

Runs locally with torch

🖥️ Hardware Requirements

Minimum:

8GB RAM

CPU support

Recommended:

16GB RAM

GPU (for faster generation)

⚠️ Notes

First run will take time (downloads model + builds index)

CPU inference may be slow

Index is stored locally in ./wiki_rag

Delete the folder if you want to rebuild the index

🧩 Future Improvements

Add PDF upload support

Increase similarity_top_k

Add chat history memory

Add streaming responses

Add multi-document support