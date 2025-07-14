import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# App title
st.title("Welcome to my Text Generator App!")

# 📄 Upload 10-Q PDF
st.header("📄 Upload your PDF File")
uploaded_file = st.file_uploader("Upload your PDF file below", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # 🧾 Load & split PDF
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Load local Q&A model
    model_name = "google/flan-t5-base"
    flan_pipe = pipeline("text2text-generation", model=model_name)
    llm = HuggingFacePipeline(pipeline=flan_pipe)

    # Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Ask user query
    query = st.text_input("Ask something like: 'What are the financial highlights?'")
    if query:
        response = qa_chain.run(query)
        st.subheader("📑 Answer")
        st.write(response[:1500]) 

