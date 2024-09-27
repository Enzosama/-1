import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
        
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings = embeddings)
    vector_store.save_local("faiss_index")
    
def get_consersational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provider context, make sure to provide all the details, if the answrer is not in provider conntext just say, "answer is not available in the context", don't provide the wrong answer. No yapping
    
    Context: \n {context}? \n
    Question: \n {question} \n
    
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0.3)
    
    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_consersational_chain()
    
    response = chain(
        {"input_documents": docs, "question" : user_question}
        , return_only_output = True)
        
    print(response)
    st.write("Reply: ", response["output_text"])
    

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini API")
    
    user_question = st.text_input("Ask a question based on your PDF Files")
    
    if user_question:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your pdf file")
        if st.button("Summit & process"):
            with st.spinner("Processing....."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                
if __name__ == "__main__":
    main()