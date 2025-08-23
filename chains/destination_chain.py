# destination_chain.py

import os
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

# -----------------------------
# Load Pinecone credentials from environment
# -----------------------------
# -----------------------------
# Initialize Pinecone client and index
# -----------------------------
pinecone_client = PineconeClient(api_key="pcsk_7BrzjA_BrMyFwP1Y6Z8oa5zgzC293Ap6mGY9nwjngjMhA6yd5ddEcQb8rDcBce5xpT3MTZ")
index = pinecone_client.index("travel-recommender-embeddings-index")

# -----------------------------
# Initialize Ollama embeddings
# -----------------------------
embedder = OllamaEmbeddings(model="llama3.2")  # Ollama embeddings object

# -----------------------------
# Pinecone retriever
# -----------------------------
def get_pinecone_retriever(k: int = 5):
    """
    Returns a LangChain Pinecone retriever compatible with Pinecone v3+ and Ollama embeddings.
    """
    vectorstore = Pinecone.from_existing_index(
        index=index,
        embedding=embedder
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# -----------------------------
# Initialize LLM and RAG chain
# -----------------------------
llm = ChatOllama(model='llama3.2')
retriever = get_pinecone_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -----------------------------
# Main function
# -----------------------------
def recommend_destinations(slots: dict):
    """
    Generate destination recommendations using RAG and Ollama embeddings.
    
    Args:
        slots: dict with keys 'trip_type', 'destination', 'budget', 'days'
    
    Returns:
        dict with 'input' and 'result'
    """
    print("Slots received:", slots)

    query = (
        f"Recommend cities, activities, hotels to stay with clear budget and some buffer "
        f"for a {slots['trip_type']} trip in {slots['destination']} within {slots['budget']} "
        f"and {slots['days']} days."
    )

    # Step 1: Retrieve documents
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs or len(retrieved_docs) == 0:
        print("⚠️ No info found in retriever — falling back to model knowledge.")
        response = llm.invoke(query)
        return {"input": query, "result": getattr(response, "content", str(response))}
    else:
        print(f"✅ Retrieved {len(retrieved_docs)} docs from retriever.")
        response = rag_chain.invoke(query)
        if hasattr(response, "content"):
            return {"input": query, "result": response.content}
        return {"input": query, "result": str(response)}
