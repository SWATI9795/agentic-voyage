# utils/prompts.py

from langchain.prompts import PromptTemplate

# --- Intent Recognition Prompt ---
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a travel assistant. Classify the user's intent from the following types:
- travel_route
- poi_recommendation
- stay_options
- general_query

Then extract any known fields:
- destination
- travel_dates
- trip_type (solo, family, adventure)
- budget

Query: {query}

Return JSON:
{
  "intent": "...",
  "destination": "...",
  "travel_dates": "...",
  "trip_type": "...",
  "budget": "..."
}
"""
)

# --- RAG: Destination/Activity Retrieval ---
destination_prompt = PromptTemplate(
    input_variables=["trip_type", "destination", "budget"],
    template="""
You are an expert travel planner.

Suggest top destinations or activities in {destination} for a {trip_type} trip within a budget of {budget}.

Return only the names and 1-line descriptions of recommended options.
"""
)

# --- Itinerary Planning ---
itinerary_prompt = PromptTemplate(
    input_variables=["destinations", "preferences"],
    template="""
Plan a detailed 3-day itinerary based on the destinations and preferences below.

Destinations:
{destinations}

User Preferences:
{preferences}

Format:
Day 1: ...
Day 2: ...
Day 3: ...
"""
)

# --- Explanation Generation ---
explanation_prompt = PromptTemplate(
    input_variables=["itinerary", "preferences"],
    template="""
You are an ethical and transparent travel assistant.

Explain the rationale behind the given itinerary using the following preferences:
- travel style
- budget
- trip goals (e.g., sightseeing, relaxation)

User Preferences:
{preferences}

Itinerary:
{itinerary}

Write a clear, 4–5 sentence explanation that builds user trust.
"""
)
