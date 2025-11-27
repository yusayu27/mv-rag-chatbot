import os
import shutil
from typing import List
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()
PERSIST_DIRECTORY = "./chroma_db"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    print(f"Loading local embedding model: {HF_EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} 
    )
    return embeddings

def build_vector_store(chunks: List[Document], clean_start: bool = True):
    if clean_start and os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    print(f"Generate embeddings for {len(chunks)} Chunks (Local)...")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Vector store successfully created in '{PERSIST_DIRECTORY}'")
    return vector_store

def load_vector_store():
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"No DB under {PERSIST_DIRECTORY} found.") 
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
    return vector_store


if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    try:
        from src.loader import load_documents
        from src.splitter import split_documents
        test_pdf = "/Users/yusuf/Desktop/personal_projects/mv-rag-chatbot/data/testpaper.pdf" # Oder data/doc.pdf
        docs = load_documents(test_pdf)
        chunks = split_documents(docs)
        build_vector_store(chunks)
    except Exception as e:
        print(f"Error: {e}")