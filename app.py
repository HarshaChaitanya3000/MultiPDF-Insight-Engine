import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Make sure the GOOGLE_APPLICATION_CREDENTIALS is set correctly
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_credentials:
    print(f"Google credentials path loaded: {google_credentials}")
else:
    st.error("Google credentials are not set in the environment. Please check your .env file.")

# Configure the Google Generative AI API with credentials
genai.configure()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.success("✅ Vector store created and saved.")

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say: "answer is not available in the context".    
    Context:
    {context}?
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if docs:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("📝 **Answer:** ", response.get("output_text", "No response generated."))
        else:
            st.warning("⚠️ No relevant documents found for your query.")
    else:
        st.error("🚨 FAISS index not found. Please process PDF documents first.")

# Main function to run the Streamlit app
def main():
    # Set up page title and icon (this must be the first Streamlit command)
    st.set_page_config(page_title="MultiPDF Insight Engine", page_icon="📂")
    
    st.title("💬 MultiPDF Insight Engine")
    st.markdown("""
    This app allows you to upload PDF documents, process them, and ask questions to extract insights from the uploaded content.
    It uses **Google Gemini** models to provide detailed and context-aware answers.
    """)
    
    st.subheader("Ask a Question:")
    user_question = st.text_input("Enter your question here:")
    
    if user_question:
        st.write("🔍 Searching for answer...")
        user_input(user_question)
    
    with st.sidebar:
        st.title("📂 Upload Your PDFs")
        st.markdown("Upload one or more PDF files to begin the process.")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process PDFs"):
            st.write("🔄 Processing PDFs...")
            if not pdf_docs:
                st.error("🚨 Please upload at least one PDF document.")
            else:
                with st.spinner("Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    else:
                        st.error("🚨 Failed to extract text from the PDF files.")

if __name__ == "__main__":
    main()
