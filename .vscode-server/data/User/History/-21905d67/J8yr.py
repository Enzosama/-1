import os
import pickle
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
    
    # Save the index
    faiss_index = vector_store.index
    vector_store.index = None
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    
    # Save the index separately
    faiss_index.save("faiss.index")
    
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, "Answer is not available in the context". Don't provide a wrong answer. No yapping.

    Context: \n {context}? \n
    Question: \n {question} \n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the store
    with open("faiss_store.pkl", "rb") as f:
        vector_store = pickle.load(f)
    
    # Load the index
    vector_store.index = FAISS.load_local("faiss.index", embeddings)
    
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

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
        if not os.path.exists("faiss_store.pkl") or not os.path.exists("faiss.index"):
            st.warning("Please process PDFs before asking questions.")
        else:
            user_input(user_question)

if __name__ == "__main__":
    main()