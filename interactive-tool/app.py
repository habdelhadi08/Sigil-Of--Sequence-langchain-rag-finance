import fitz
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="File Analyzer", layout="wide")
st.title("📄 Welcome to My File Analyzer App")

# Define available LLM options for selection
LLM_OPTIONS = {
    "Flan-T5 Small": "google/flan-t5-small",
    "Flan-T5 Base": "google/flan-t5-base"
}

# Sidebar model selector
model_choice = st.sidebar.selectbox("Choose a language model", list(LLM_OPTIONS.keys()))

@st.cache_resource
def load_llm(model_name):
    st.info(f"Using model: `{model_name}`")
    pipe = pipeline("text2text-generation", model=model_name)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

@st.cache_resource
def build_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return Chroma.from_documents(_docs, embeddings)

# Upload PDF file
st.header("Step 1: Upload Your PDF File")
uploaded_file = st.file_uploader("Upload your PDF file below", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded successfully!")

    # Process PDF
    with st.spinner("📂 Processing document..."):
        docs = process_pdf("temp.pdf")

    # Build vectorstore
    with st.spinner("🔍 Building Chroma Vectorstore..."):
        vectorstore = build_vectorstore(docs)

    # Load selected LLM
    with st.spinner("🤖 Loading your selected model..."):
        llm = load_llm(LLM_OPTIONS[model_choice])

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Query input
    st.header("Step 2: Ask Questions from the Document")
    user_question = st.text_input("🔎 Enter your question:", placeholder="e.g., What are the financial highlights?")

    if user_question:
        with st.spinner("💬 Generating response..."):
            response = qa_chain.run(user_question)
        st.subheader("📑 Answer")
        st.write(response[:1500])

