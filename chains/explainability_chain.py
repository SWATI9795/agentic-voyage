# chains/explainability_chain.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFaceHub

# Initialize local LLM via Ollama (e.g., llama3)
#llm = ChatOllama(model="llama3.2", temperature=0)

HF_TOKEN = os.getenv("HFACE_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-13b-chat-hf",   # Change to 13B if you want larger
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0}
)

# Prompt template for generating explainable reasons in bullet points
explanation_prompt = PromptTemplate(
    input_variables=["itinerary", "preferences"],
    template="""
You are an ethical travel assistant. Based on the following user preferences and itinerary, 
explain why these destinations and activities were chosen.

Guidelines:
- Respond ONLY in bullet points (use '-' at the start of each line).
- Keep each point concise and clear.
- Cover: budget, activity type, travel style (solo/family/adventure/etc.), proximity, uniqueness or cultural value.

User Preferences:
{preferences}

Itinerary:
{itinerary}

Explanation (in bullet points):
- 
"""
)

explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

def generate_explanation(itinerary: dict, preferences: dict) -> str:
    """Generate bullet-point style explanation for itinerary choices."""

    # Convert itinerary dict into readable string for the prompt
    if isinstance(itinerary, dict):
        itinerary_str = ""
        for day, details in itinerary.items():
            itinerary_str += f"\n{day.title()}:\n"
            if isinstance(details, dict):
                if "activities" in details:
                    itinerary_str += "  Activities: " + ", ".join(details["activities"]) + "\n"
                if "stay" in details:
                    itinerary_str += f"  Stay: {details['stay']}\n"
                if "description" in details:
                    itinerary_str += f"  About: {details['description']}\n"
            else:
                itinerary_str += f"  {details}\n"
    else:
        itinerary_str = str(itinerary)

    # Format preferences nicely
    prefs_str = ", ".join([f"{k}: {v}" for k, v in preferences.items()])

    # Invoke LLM
    response = explanation_chain.invoke({
        "itinerary": itinerary_str,
        "preferences": prefs_str
    })

    # Extract clean string (response can be dict depending on LangChain version)
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    elif hasattr(response, "content"):  # sometimes returns AIMessage
        return response.content.strip()
    return str(response).strip()
