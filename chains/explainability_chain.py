# chains/explainability_chain.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Initialize local LLM via Ollama (e.g., llama3)
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Prompt template for generating explainable reasons
explanation_prompt = PromptTemplate(
    input_variables=["itinerary", "preferences"],
    template="""
You are an ethical travel assistant. Based on the following user preferences and itinerary, explain why these destinations and activities were chosen.

Be concise and justify each element of the plan in terms of:
- budget
- activity type
- travel style (solo/family/adventure/etc.)
- proximity
- uniqueness or cultural value

User Preferences:
{preferences}

Itinerary:
{itinerary}

Explanation:
"""
)

explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

def generate_explanation(itinerary: str, preferences: dict) -> str:
    # Prepare preference string
    prefs_str = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
    return explanation_chain.invoke({
        "itinerary": itinerary,
        "preferences": prefs_str
    })
