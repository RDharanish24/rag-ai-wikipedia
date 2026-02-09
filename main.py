import os
import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
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

@st.cache_resource
def get_index():
    if os.path.exists(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    reader = WikipediaReader()
    documents = reader.load_data(pages=PAGES, auto_suggest=False)

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )

    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()

    llm = HuggingFaceLLM(
        model_name="google/flan-t5-base",
        tokenizer_name="google/flan-t5-base",
        context_window=512,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0}
    )

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=3
    )

def main():
    st.title("🆓 Wikipedia RAG (Fully Free, No Ollama)")

    question = st.text_input("Ask a question about AI / ML:")

    if st.button("Submit") and question:
        with st.spinner("Thinking..."):
            query_engine = get_query_engine()
            response = query_engine.query(question)

            st.subheader("Answer")
            st.write(response.response)

            st.subheader("Retrieved Context")
            for node in response.source_nodes:
                st.markdown(node.get_content())

if __name__ == "__main__":
    main()
