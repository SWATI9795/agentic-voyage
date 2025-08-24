import os
from dotenv import load_dotenv
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.llms import HuggingFaceHub


#load_dotenv()

#HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#llm = ChatOllama(model="llama3.2", temperature=0)

# Initialize LLaMA 2 Chat model
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-13b-chat-hf",   # Change to 13B if you want larger
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0}
)

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a travel assistant. Classify the user's intent (destination_info / activity / budget / general)
    and extract the following fields from their query:

    - intent (one of: destination_info, activity, budget, general)
    - destination (e.g., Goa, Udaipur)
    - budget (low / moderate / luxury or a ₹ range)
    - trip_type (e.g., honeymoon, adventure, family)
    - days (number of days mentioned, e.g., 3, 5. If not mentioned, guess based on query)

    Output must be valid JSON like this:
    {{
      "intent": "...",
      "destination": "...",
      "budget": "...",
      "trip_type": "...",
      "days": "..."
    }}

    If any field is not clearly stated, make a best guess.

    Respond ONLY with JSON. Do NOT include any text outside this format.

    Query: {query}
    """
)

intent_chain = prompt | llm | StrOutputParser()


def get_intent_and_slots(query):
    response = intent_chain.invoke({"query": query})

    print(f"response:", response)

    # Safely extract response string
    response_text = response.text if hasattr(response, "text") else str(response)

    # Try to parse JSON from model response
    try:
        parsed = json.loads(response_text)
        intent = parsed.get("intent", "recommend")
        slots = {
            "destination": parsed.get("destination", "India"),
            "budget": parsed.get("budget", "moderate"),
            "trip_type": parsed.get("trip_type", "general"),
            "days": parsed.get("days", "3")  # Default to 3 if missing
        }
    except json.JSONDecodeError:
        print("❌ Could not parse JSON — fallback used.")
        intent = "recommend"
        slots = {
            "destination": "India",
            "budget": "moderate",
            "trip_type": "general",
            "days": "3"
        }
    return intent, slots