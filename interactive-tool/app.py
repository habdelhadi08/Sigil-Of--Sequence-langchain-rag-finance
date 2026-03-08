import fitz
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="📄 File Analyzer", layout="wide")
st.title("📄 Welcome to My File Analyzer App")

# -----------------------------
# Sidebar: Model & Splitter Settings
# -----------------------------
LLM_OPTIONS = {
    "Flan-T5 base": "google/flan-t5-base",
    "Flan-T5 Large": "google/flan-t5-large"
}

mode = st.sidebar.radio("Mode", ["Single Model", "Compare Two Models"])

if mode == "Single Model":
    model_choice = st.sidebar.selectbox("Choose a language model", list(LLM_OPTIONS.keys()))
else:
    model_choice_1 = st.sidebar.selectbox("Model 1", list(LLM_OPTIONS.keys()), index=0)
    model_choice_2 = st.sidebar.selectbox("Model 2", list(LLM_OPTIONS.keys()), index=1)

# Text splitter settings
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)

# -----------------------------
# Functions
# -----------------------------
@st.cache_resource
def load_llm(model_name):
    pipe = pipeline("text2text-generation", model=model_name, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def process_pdf(file_path, chunk_size, chunk_overlap):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

@st.cache_resource
def build_vectorstore(_docs):  # Note the leading underscore to avoid hashing
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return Chroma.from_documents(_docs, embeddings)

# -----------------------------
# Step 1: Upload PDF
# -----------------------------
st.header("Step 1: Upload Your PDF File")
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ File uploaded successfully!")

    with st.spinner("📂 Processing document..."):
        docs = process_pdf(tmp_path, chunk_size, chunk_overlap)

    if docs:
        with st.spinner("🔍 Building Chroma Vectorstore..."):
            vectorstore = build_vectorstore(docs)

        # -----------------------------
        # Load LLM(s)
        # -----------------------------
        if mode == "Single Model":
            llm = load_llm(LLM_OPTIONS[model_choice])
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        else:
            llm1 = load_llm(LLM_OPTIONS[model_choice_1])
            llm2 = load_llm(LLM_OPTIONS[model_choice_2])
            qa_chain_1 = RetrievalQA.from_chain_type(llm=llm1, retriever=vectorstore.as_retriever())
            qa_chain_2 = RetrievalQA.from_chain_type(llm=llm2, retriever=vectorstore.as_retriever())

        # -----------------------------
        # Step 2: Ask Questions
        # -----------------------------
        st.header("Step 2: Ask Questions from the Document")
        user_question = st.text_input("🔎 Enter your question:", placeholder="e.g., What are the financial highlights?")

        if user_question:
            with st.spinner("💬 Generating response..."):
                if mode == "Single Model":
                    response = qa_chain.run(user_question)
                    st.subheader("📑 Answer")
                    st.text_area("Answer", value=response, height=300)
                else:
                    response1 = qa_chain_1.run(user_question)
                    response2 = qa_chain_2.run(user_question)
                    st.subheader("📑 Answers Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{model_choice_1}**")
                        st.text_area("Answer", value=response1, height=300)
                    with col2:
                        st.markdown(f"**{model_choice_2}**")
                        st.text_area("Answer", value=response2, height=300)
    else:
        st.warning("No documents were extracted from the PDF.")
