from langchain.chains import RetrievalQA
from utils.load_vectorstore import get_pinecone_retriever
from langchain_ollama import OllamaLLM, ChatOllama

llm = ChatOllama(model='llama3.2')
retriever = get_pinecone_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=ChatOllama(model='llama3.2'), retriever=retriever)

def recommend_destinations(slots):
    print(slots)
    #return rag_chain.invoke(f"Recommend cities and activities for a {slots['trip_type']} trip in {slots['destination']} within {slots['budget']}")
    print("Slots received:", slots)

    query = f"Recommend cities, activities, hotels to stay with clear budget and keep some buffer for a {slots['trip_type']} trip in {slots['destination']} within {slots['budget']} and {slots['days']} days."

    # Step 1: Try retriever directly
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs or len(retrieved_docs) == 0:
        print("⚠️ No info found in retriever — falling back to model knowledge.")
        response = llm.invoke(query)
        return {"input": query, "result": response.content if hasattr(response, "content") else str(response)}
    else:
        print(f"✅ Retrieved {len(retrieved_docs)} docs from retriever.")
        response = rag_chain.invoke(query)
        return response