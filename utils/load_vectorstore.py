import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# ✅ Correct v3.x client init
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(INDEX_NAME)

# Set up embedder
embedder = OllamaEmbeddings(model="mxbai-embed-large")

MAX_PINECONE_PAYLOAD = 4 * 1024 * 1024  # 4MB
BATCH_SIZE = 50

def get_pinecone_retriever():
    return LangchainPinecone(index, embedder.embed_query, "text").as_retriever()

def embed_pdf_to_pinecone():
    loader = UnstructuredPDFLoader("data/Travel-Data-for-Model-Training.pdf")
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

embed_pdf_to_pinecone()
