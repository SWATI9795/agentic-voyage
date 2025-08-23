from langchain.chains import RetrievalQA
from utils.load_vectorstore import get_pinecone_retriever
from langchain_ollama import ChatOllama

# Initialize LLM once
llm = ChatOllama(model='llama3.2')

# Get Pinecone v3-compatible retriever
retriever = get_pinecone_retriever()

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def recommend_destinations(slots):
    """
    Generate destination recommendations using RAG.
    
    slots: dict with keys 'trip_type', 'destination', 'budget', 'days'
    """
    print("Slots received:", slots)

    # Construct query
    query = (
        f"Recommend cities, activities, hotels to stay with clear budget and keep some buffer "
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
        # response can be string or object, depending on LangChain version
        if hasattr(response, "content"):
            return {"input": query, "result": response.content}
        return {"input": query, "result": str(response)}
