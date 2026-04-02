RAG-DOC-Q-A-Chatbot-with-NVIDIA-DeepSeek

RAG (Retrieval-Augmented Generation) powered Document Q/A Chatbot that lets you ask questions directly from your PDFs using NVIDIA DeepSeek — an advanced Large Language Model hosted on NVIDIA AI Endpoints.

Project Overview

This Streamlit-based chatbot combines document retrieval and LLM reasoning to deliver accurate, context-aware answers from your own files.
It uses FAISS vector database for efficient similarity search and NVIDIA Embeddings for generating semantic representations of your document text.

Key Features

PDF Loader: Upload or load multiple PDF files at once.
NVIDIA DeepSeek Model: Smart, accurate, and context-sensitive responses powered by deepseek-ai/deepseek-v3.1-terminus.
Vector Database (FAISS): Quickly retrieves the most relevant text chunks from your PDFs.
RAG Pipeline: Combines retrieval + generation to provide precise answers.
Response Timer: Measures how fast the model generates responses.
Similarity Viewer: Expander section showing retrieved document chunks used in the answer.

Tech Stack
Component	Technology
LLM	NVIDIA DeepSeek via LangChain
Embeddings	NVIDIAEmbeddings
 Document Loader	PyPDFDirectoryLoader
 Vector Store	FAISS
 Framework	Streamlit
 LangChain Components	RetrievalQA, PromptTemplate
 
 Workflow

1️ Load PDFs from a folder or upload directly.
2️ Split text into chunks using RecursiveCharacterTextSplitter.
3️⃣ Create embeddings with NVIDIAEmbeddings.
4️⃣ Store vectors in FAISS.
5️⃣ Use RetrievalQA to retrieve relevant context and generate an answer.
6️⃣ Display response and show the supporting document chunks.

 How It Works (RAG Pipeline)
 PDFs ➜ Split ➜ Embeddings ➜ FAISS Retrieval ➜ NVIDIA LLM ➜ Final Answer


 Retrieval: Finds the most relevant chunks.
 Augmentation: Adds them as context.
 Generation: LLM (DeepSeek) produces an informed answer.

 Usage

1️ Place your PDF files inside a folder named “SIH”
2️ Run the Streamlit app:

streamlit run app.py


3️ Click “ Create Document Embeddings”
4️ Ask any question in the text box 
5️ View the result + source context inside the expander section.
