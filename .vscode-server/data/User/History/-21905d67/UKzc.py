import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    if not text.strip():
        return []  # Return an empty list if there's no text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("No text chunks to process. The PDF might be empty or unreadable.")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini API")

    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
    
    if pdf_docs:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.warning("The uploaded PDF(s) appear to be empty or unreadable.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.success("PDFs processed successfully!")
                    else:
                        st.warning("Failed to create vector store. Please try again with different PDFs.")

    user_question = st.text_input("Ask a question based on your PDF files")
    
    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please process PDFs before asking questions.")
        else:
            user_input(user_question)

if __name__ == "__main__":
    main()