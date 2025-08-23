import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.3)

# Prompt
prompt = PromptTemplate(
    input_variables=["input", "days"],
    template="""
You are a helpful travel planner.

Generate a detailed travel itinerary for {days} days ONLY for the given destination {destinations} in the user query.
Do NOT suggest unrelated or additional cities unless explicitly mentioned by the user.

Respond ONLY in valid JSON format like this:
{{
  "day_1": {{
    "activities": ["Activity 1", "Activity 2"],
    "stay": "Suggested stay",
    "description": "Short description of the place"
  }},
  "day_2": {{
    "activities": ["Activity 1", "Activity 2"],
    "stay": "Suggested stay",
    "description": "Short description of the place"
  }}
}}

User Query: {input}
"""
)

itinerary_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=StrOutputParser()
)

def generate_itinerary(destinations, slots):
    days = slots.get("days", 3)  # default to 3
    response = itinerary_chain.invoke(
        {"input": f"{destinations}, {slots}", "days": days, "destinations": destinations}
    )

    # Debug print
    #print("Raw itinerary_chain response:", response)

    # Handle dict outputs (LangChain sometimes wraps in {"text": ...})
    if isinstance(response, dict):
        response_text = response.get("text") or response.get("output") or str(response)
    else:
        response_text = str(response)

    # Try parsing JSON
    try:
        parsed = json.loads(response_text)
        print("✅ Parsed itinerary JSON")
        return parsed
    except Exception as e:
        print("⚠️ Could not parse JSON:", e)
        return {"raw_itinerary": response_text}
