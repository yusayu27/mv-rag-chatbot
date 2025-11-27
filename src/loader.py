import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_documents(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {file_path}")
    return docs


if __name__ == "__main__":
    test_path = "/Users/yusuf/Desktop/personal_projects/mv-rag-chatbot/data/testpaper.pdf"
    documents=load_documents(test_path)
    print(f"First document content preview: {documents[0].page_content[:200]}")
    print(f"First document metadata: {documents[0].metadata}")

