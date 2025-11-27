import streamlit as st
import os
import sys
sys.path.append(os.getcwd())

try:
    from src.rag_chain import answer_question, create_retriever, get_llm
    from src.embeddings_store import load_vector_store
except ImportError as e:
    st.error(f"Import Error: {e}. Please run 'streamlit run src/app.py' from the root directory.")
    st.stop()

st.set_page_config(page_title="RAG Chatbot Demo", page_icon="🤖", layout="centered")

st.title("Document Chatbot")
st.caption("Hybrid RAG: Local Embeddings (CPU) + Google Gemini Flash (Cloud)")

with st.sidebar:
    st.header(" Setup Info")
    st.success("Embeddings: Sentence-Transformers (Local)")
    st.success("LLM: Gemini 2.5 Flash (Cloud)")
    
    st.markdown("---")
    st.markdown("### About the Project")
    st.info(
        "Demonstrating a privacy-first Hybrid RAG architecture. "
        "Vectorization and retrieval run locally. "
        "Only relevant context chunks are transmitted for LLM inference, ensuring data minimization."
    )
    
    if st.button("Reset memory "):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def initialize_rag_system():
    try:
        vector_store = load_vector_store()
        retriever = create_retriever(vector_store)
        llm = get_llm()
        return retriever, llm
    except Exception as e:
        return None, str(e)

retriever, llm = initialize_rag_system()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I analyzed the documents. How can I assist you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Your questions about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing document..."):
            try:
                relevant_docs = retriever.invoke(prompt)
                full_response = answer_question(prompt, retriever, llm)
                message_placeholder.markdown(full_response)
                with st.expander("Show used document sources"):
                    if relevant_docs:
                        for i, doc in enumerate(relevant_docs):
                            page_num = doc.metadata.get('page', 'Unknown')
                            source_preview = doc.page_content[:150].replace("\n", " ")
                            st.markdown(f"**Source {i+1}** (Page {page_num}):")
                            st.caption(f"...{source_preview}...")
                    else:
                        st.info("No specific document parts found.")
                        
            except Exception as e:
                st.error(f"An error occured: {e}")
                full_response = "Sorry, there was a system error."

    st.session_state.messages.append({"role": "assistant", "content": full_response})