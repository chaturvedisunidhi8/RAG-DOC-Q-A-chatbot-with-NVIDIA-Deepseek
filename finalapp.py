import streamlit as st 
import os 
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Streamlit title
st.title("📘 RAG Document Q/A Chatbot with NVIDIA DeepSeek")

# Initialize LLM
llm = ChatNVIDIA(model="deepseek-ai/deepseek-v3.1-terminus")

# Function to create vector embeddings from PDFs
def vector_embeddings():
    if "vectors" not in st.session_state:
        with st.spinner("📄 Loading PDFs and creating embeddings..."):
            # Initialize NVIDIA embeddings
            st.session_state.embeddings = NVIDIAEmbeddings()
            
            # Load PDFs from folder "SIH"
            st.session_state.loader = PyPDFLoader("SIH.pdf")
            st.session_state.docs = st.session_state.loader.load()

            if not st.session_state.docs:
                st.warning("⚠️ No PDFs found in the 'SIH' folder!")
                return
            
            # Split text into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
        st.success("✅ FAISS vector store created successfully using NVIDIA embeddings!")

# ✅ Correct prompt template — note {context} and {question}
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the context below. 
If you don't know the answer, just say that you don't know.
Please answer accurately and concisely.

Context:
{context}

Question:
{question}
"""
)

# User input field
prompt1 = st.text_input("💬 Enter your question from documents:")

# Button to create embeddings
if st.button("⚙️ Create Document Embeddings"):
    vector_embeddings()

# When user enters a question
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create embeddings first by clicking '⚙️ Create Document Embeddings'.")
    else:
        retriever = st.session_state.vectors.as_retriever()

        # ✅ Create RetrievalQA chain with correct variables
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        # Query the model
        start = time.process_time()
        response = qa_chain({"query": prompt1})
        elapsed_time = time.process_time() - start

        # Display answer
        st.subheader("🧠 Answer:")
        st.write(response["result"])
        st.write(f"⏱ Response Time: {elapsed_time:.2f} seconds")

        # Show retrieved chunks
        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
