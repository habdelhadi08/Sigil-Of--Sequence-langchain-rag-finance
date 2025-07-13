# interactive-tool/app.py

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

st.title("🧠 10-Q Report Insight Generator")

uploaded_file = st.file_uploader("Upload 10-Q PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=vectorstore.as_retriever())

    query = st.text_input("Ask something like: 'What are the financial highlights?'")
    if query:
        response = qa_chain.run(query)
        st.write(response)
