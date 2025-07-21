from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Define the schema as a list of ResponseSchema objects
response_schemas = [
    ResponseSchema(name="day_1", description="Plan for Day 1"),
    ResponseSchema(name="day_2", description="Plan for Day 2"),
    ResponseSchema(name="day_3", description="Plan for Day 3"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a helpful travel planner.

    Generate a 3-day travel itinerary based on the user's preferences.
    
    Respond ONLY in this JSON format:
    {{
      "day_1": "<activities for day 1>",
      "day_2": "<activities for day 2>",
      "day_3": "<activities for day 3>"
    }}
    
    User Query: {input}
    """
)
itinerary_chain = LLMChain(llm=ChatOllama(model="llama3.2"), prompt=prompt, output_parser=parser)

def generate_itinerary(destinations, slots):
    return itinerary_chain.invoke({"input": f"{destinations}, {slots}"})