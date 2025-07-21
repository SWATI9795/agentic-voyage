import streamlit as st
from chains.intent_chain import get_intent_and_slots
from chains.destination_chain import recommend_destinations
from chains.itinerary_chain import generate_itinerary
from chains.explainability_chain import generate_explanation
from guards.guardrails import apply_guardrails
from utils.format_output import format_response

st.title("✈️ AI Travel Recommender")
query = st.text_input("Where do you want to go and what do you like?")

if query:
    with st.spinner("Planning your trip..."):
        intent, slots = get_intent_and_slots(query)
        print("Intent and slots extracted...")
        destination_result = recommend_destinations(slots)
        print("Destination result extracted...")
        itinerary = generate_itinerary(destination_result, slots)
        print("Itinerary result generated...")
        explanation = generate_explanation(itinerary, destination_result)
        print("Explanation generated...")
        final_output = apply_guardrails(itinerary, explanation, slots)
        print("Guardrails applied...")
        st.markdown(format_response(final_output))

#if __name__ == "__main__":
#    from utils.load_vectorstore import embed_pdf_to_pinecone
#    embed_pdf_to_pinecone()
