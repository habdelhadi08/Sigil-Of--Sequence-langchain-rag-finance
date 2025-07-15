import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="File Analyzer", layout="wide")
st.title("📄 Welcome to My File Analyzer App")

# Caching heavy steps
@st.cache_resource
def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

@st.cache_resource
def build_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return FAISS.from_documents(_docs, embeddings)

@st.cache_resource
def load_llm():
    flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    return HuggingFacePipeline(pipeline=flan_pipe)

# 📄 Upload 10-Q PDF
st.header("Upload your PDF File")
uploaded_file = st.file_uploader("Upload your PDF file below", type="pdf")

if uploaded_file:
    # Save file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Step 1: Process and Split
    with st.spinner("📂 Processing PDF..."):
        docs = process_pdf("temp.pdf")

    # Step 2: Embedding & Vectorstore
    with st.spinner("🔍 Creating FAISS Index..."):
        vectorstore = build_vectorstore(docs)

    # Step 3: Load the LLM
    with st.spinner("🤖 Loading Q&A Model..."):
        llm = load_llm()

    # Step 4: Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Step 5: User query input
    query = st.text_input("🔎 Ask your question:", placeholder="e.g. What are the financial highlights?")
    if query:
        with st.spinner("💬 Generating answer..."):
            response = qa_chain.run(query)
        st.subheader("📑 Answer")
        st.write(response[:1500])
