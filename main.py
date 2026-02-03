import os
import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama