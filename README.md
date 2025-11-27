# Hybrid RAG Chatbot 

I developed this Hybrid RAG Application as a fun weekend project.
It is designed so that organizations with strict privacy requirements (like automotive R&D or legal) can perform semantic search and Q&A on internal documents without exposing the full dataset to the cloud.
It indexes data locally on the CPU and only uses the Cloud for the final reasoning step.

## Technologies

- Python 3.10+
- LangChain (Community & Core)
- pypdf, RecursiveCharacterTextSplitter
- sentence-transformers (HuggingFace) via all-MiniLM-L6-v2
- ChromaDB (Local Persistence)
- Google Gemini 2.5 Flash
- Streamlit


## Requirements


- Python >= Version 3.10
- Google AI Studio API Key 
- PDF Documents for analysis 


## Setup


Clone the Project

### 1. Clone the project

### 2. Environment Setup
-Check if Python is installed:
```bash
python --version
```


### 3. Create a virtual environment in the Root Directory
```bash
python3 -m venv .venv
```


### 4.Install Dependencies
 -Install the required Libraries.



## Configuration
Create a .env file in the root directory and add your API Key:
```bash
GOOGLE_API_KEY=your_key_here
```


## Getting the Results

## 1. Data Ingestion:
Put your target PDF into data/testpaper.pdf.
Run the ingestion script to split the text and build the local Vector Store:
```bash
python -m src.embeddings_store
````

## 2. Launch Web App:
```bash
streamlit run src/app.py
```

You should be automatically redirected to the Web App. If not, simply click the link in the Terminal.

## Challenges Faced and how I solved them
-Running embeddings in the cloud caused rate limits and slow responses. I solved this problem by moving embeddings locally with sentence transformers.

-Different LangChain/HF packages conflicted. Pinning exact versions in requirements.txt solved this.
