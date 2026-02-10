import os
import streamlit as st
import torch
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# Configuration
INDEX_DIR = "./wiki_rag"
PAGES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Convolutional neural network",
    "Long short-term memory"
]

# --- 1. SETUP GLOBAL SETTINGS (Crucial Step) ---
# We set these globally so they are used for BOTH creating and loading the index.
@st.cache_resource
def setup_settings():
    # 1. Embedding Model (Keep this)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. LLM (Switch to Qwen 2.5 - 0.5B Instruct)
    # This model is "Decoder-only", so it works instantly with HuggingFaceLLM
    Settings.llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
        context_window=2048,   # Much larger context than T5!
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.3, "do_sample": True},
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float32} 
    )
    
    # 3. Chunk Size
    # Qwen has a larger context window, so we can use standard chunk sizes
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50


setup_settings()

@st.cache_resource
def get_index():
    if os.path.exists(INDEX_DIR):
        # When loading, it now uses the Settings.embed_model automatically
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    # If creating new
    reader = WikipediaReader()
    documents = reader.load_data(pages=PAGES, auto_suggest=False)

    # Create index (uses Settings.embed_model and Settings.chunk_size automatically)
    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

def main():
    st.title("🆓 Wikipedia RAG (Flan-T5 + Local)")

    question = st.text_input("Ask a question about AI / ML:")

    if st.button("Submit") and question:
        with st.spinner("Thinking... (This may be slow on CPU)"):
            try:
                index = get_index()
                # similarity_top_k=1 because our context window is tiny (512 tokens)
                # If we retrieve 3 chunks, we will overflow the model context.
                query_engine = index.as_query_engine(similarity_top_k=1)
                
                response = query_engine.query(question)

                st.subheader("Answer")
                st.write(response.response)

                st.subheader("Retrieved Context")
                for node in response.source_nodes:
                    st.info(node.get_content())
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()