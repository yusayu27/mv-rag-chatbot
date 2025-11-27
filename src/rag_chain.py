import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from src.embeddings_store import load_vector_store
except ImportError:
        from embeddings_store import load_vector_store

load_dotenv()

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        max_retries=2,
    )
    return llm

def create_retriever(vector_store, k: int = 6):
    return vector_store.as_retriever(search_kwargs={"k": k})

def answer_question(question: str, retriever, llm) -> str:
    print(f"Searching context for: '{question}'")
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return "I don't know based on the provided documents."
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    system_prompt = """You are a helpful research assistant. 
    Use the provided context to answer the user's question.
    If the context contains the answer, explain it clearly.
    If the context mentions related topics but not the exact answer, summarize what is available.
    If the answer is completely missing, say "I don't know based on the provided context".
    Context:
    {context}
    Question: {question} """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    return response

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    try:
        print("--- RAG Chain Start ---")
        vector_store = load_vector_store()
        retriever = create_retriever(vector_store, k=6)
        llm = get_llm()
        question= "What is accurate capability measurement?"
        print(f"Answer: {answer_question(question, retriever, llm)}")
    except Exception as e:
        print(f"Error: {e}")