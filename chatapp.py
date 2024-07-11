import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "Answer not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":robot_face:")
    
    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False

    if st.session_state['dark_mode']:
        st.markdown(
            """
            <style>
                .main {
                    background-color: black;
                    color: white;
                }
                .stTextInput > div > div > input {
                    background-color: #333333;
                    color: white;
                }
                .stTextInput > div > label {
                    color: white;
                }
                .stButton > button {
                    background-color: #333333;
                    color: white;
                }
                .stSpinner {
                    color: white;
                }
                h2, h3, h4, h5, h6 {
                    color: white;
                }
                
                .footer {
                    background-color: #262730;
                    color: white;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
                .main {
                    background-color: white;
                    color: black;
                }
                .stTextInput > div > div > input {
                    background-color: #f1f1f1;
                    color: black;
                }
                .stButton > button {
                    background-color: #f1f1f1;
                    color: black;
                }
                .footer {
                    background-color: #f1f1f1;
                    color: black;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.sidebar.title("PDF Chatbot")
    st.sidebar.image("img/robot.jpg", use_column_width=False, width=280)
    st.sidebar.markdown("---")
    st.sidebar.title("Upload PDF Files")
    pdf_docs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("Processing complete!")

    st.title("PDF Chatbot")
    st.write("Ask questions based on the content of the uploaded PDF files.")
    
    user_question = st.text_input("Enter your question here:")
    if user_question:
        user_input(user_question)

    st.sidebar.markdown("---")
    st.sidebar.image("img/creator.jpg", use_column_width=True)
    st.sidebar.write("AI App created by Vinayak Bhatia")

    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                padding: 10px;
                
            }
        </style>
        <div class="footer">
            © <a href="https://github.com/vvinayakkk" target="_blank" style="color: inherit;">Vinayak Bhatia</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

    toggle_button = st.button("🌓")
    if toggle_button:
        st.session_state['dark_mode'] = not st.session_state['dark_mode']
        st.rerun()

if __name__ == "__main__":
    main()
