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
    "Flan-T5 Small": "google/flan-t5-small",
    "Flan-T5 Base": "google/flan-t5-base"
}
model_choice = st.sidebar.selectbox("Choose a language model", list(LLM_OPTIONS.keys()))

chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50)

# -----------------------------
# Functions
# -----------------------------
@st.cache_resource
def load_llm(model_name):
    st.info(f"Using model: `{model_name}`")
    pipe = pipeline("text2text-generation", model=model_name, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def process_pdf(file_path, chunk_size, chunk_overlap):
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

@st.cache_resource
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return Chroma.from_documents(docs, embeddings)

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

        with st.spinner("🤖 Loading your selected model..."):
            llm = load_llm(LLM_OPTIONS[model_choice])

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        # -----------------------------
        # Step 2: Ask Questions
        # -----------------------------
        st.header("Step 2: Ask Questions from the Document")
        user_question = st.text_input("🔎 Enter your question:", placeholder="e.g., What are the financial highlights?")

        if user_question:
            with st.spinner("💬 Generating response..."):
                try:
                    response = qa_chain.run(user_question)
                    st.subheader("📑 Answer")
                    st.text_area("Answer", value=response, height=300)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.warning("No documents were extracted from the PDF.")

