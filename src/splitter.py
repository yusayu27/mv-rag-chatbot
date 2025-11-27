import sys
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(docs: List[Document], chunk_size: int = 600, chunk_overlap: int = 120) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


if __name__ == "__main__":
    from loader import load_documents 
    sys.path.append(os.getcwd()) 
    test_path = "/Users/yusuf/Desktop/personal_projects/mv-rag-chatbot/data/testpaper.pdf"
    original_docs = load_documents(test_path)        
    chunks = split_documents(original_docs, chunk_size=800, chunk_overlap=150)
    print(f"\n Analyzing the first 3 chunks")
    for i, chunk in enumerate(chunks[:3]):
        content_preview = chunk.page_content.replace('\n', ' ')[:100]
        print(f" Chunk {i+1} (Länge: {len(chunk.page_content)}):")
        print(f" Content: {content_preview}...")
        print(f" Metadata: {chunk.metadata}") 