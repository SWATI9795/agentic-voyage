from langchain.chains import RetrievalQA
from utils.load_vectorstore import get_pinecone_retriever
from langchain_ollama import OllamaLLM, ChatOllama

retriever = get_pinecone_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=ChatOllama(model='llama3.2'), retriever=retriever)

def recommend_destinations(slots):
    print(slots)
    return rag_chain.invoke(f"Recommend cities and activities for a {slots['trip_type']} trip in {slots['destination']} within {slots['budget']}")