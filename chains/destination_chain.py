import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from utils.load_vectorstore import get_pinecone_retriever
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_community.llms import HuggingFaceHub

#load_dotenv()
#HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-13b-chat-hf",   # Change to 13B if you want larger
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0}
)

retriever = get_pinecone_retriever()
rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def recommend_destinations(slots):
    print(slots)
    return rag_chain.invoke(f"Recommend cities and activities for a {slots['trip_type']} trip in {slots['destination']} within {slots['budget']}")