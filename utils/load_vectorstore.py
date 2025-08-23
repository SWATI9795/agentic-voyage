# utils/load_vectorstore.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, Index
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")  # optional default

if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("Please set PINECONE_API_KEY and INDEX_NAME in your environment variables.")

# -----------------------------
# Initialize Pinecone client (v3)
# -----------------------------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Access existing index (v3 style)
index = Index(name=INDEX_NAME)

# -----------------------------
# Embedder
# -----------------------------
embedder = OllamaEmbeddings(model="mxbai-embed-large")

# -----------------------------
# Retriever
# -----------------------------
def get_pinecone_retriever(k: int = 5):
    """
    Returns a Pinecone retriever compatible with Python 3.13 / Pinecone v3.
    """
    vectorstore = LangchainPinecone.from_existing_index(index=index, embedding=embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# -----------------------------
# PDF to Pinecone
# -----------------------------
MAX_PINECONE_PAYLOAD = 4 * 1024 * 1024  # 4MB
BATCH_SIZE = 50

def embed_pdf_to_pinecone(pdf_path="data/Travel-Data-for-Model-Training.pdf"):
    """
    Load PDF as text-only (strategy='fast'), split, and upload to Pinecone.
    This avoids OpenCV / libGL issues on Streamlit Cloud.
    """
    loader = UnstructuredPDFLoader(pdf_path, strategy="fast")  # text-only
    raw_documents = loader.load()
    print(f"✅ Loaded {len(raw_documents)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(raw_documents)
    print(f"📦 Split into {len(chunks)} chunks")

    valid_chunks = [
        doc for doc in chunks
        if len(doc.page_content.encode("utf-8")) < MAX_PINECONE_PAYLOAD
    ]
    print(f"✅ {len(valid_chunks)} chunks fit within Pinecone size limits")

    for i in range(0, len(valid_chunks), BATCH_SIZE):
        batch = valid_chunks[i:i + BATCH_SIZE]
        print(f"🚀 Uploading batch {i//BATCH_SIZE + 1} with {len(batch)} chunks...")
        LangchainPinecone.from_documents(batch, embedding=embedder, index_name=INDEX_NAME)

    print("✅ All chunks uploaded to Pinecone")
