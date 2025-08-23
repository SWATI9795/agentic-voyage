import streamlit as st
from chains.intent_chain import get_intent_and_slots
from chains.destination_chain import recommend_destinations
from chains.itinerary_chain import generate_itinerary
from chains.explainability_chain import generate_explanation
from guards.guardrails import input_guardrails, output_guardrails
from utils.format_output import format_response
from utils.evaluate_response import evaluate_response
from utils.load_vectorstore import embed_pdf_to_pinecone


# -------------------------
# Initialize session state
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "slots" not in st.session_state:
    st.session_state.slots = {
        "destination": None,
        "trip_type": None,
        "budget": None,
        "days": None
    }

# -------------------------
# Helper: update slots
# -------------------------
def update_slots(new_slots: dict):
    """Merge new slots with stored slots in session_state"""
    for key, value in new_slots.items():
        if value and str(value).strip():
            st.session_state.slots[key] = value
    return st.session_state.slots

# -------------------------
# Chat bubble renderer
# -------------------------
def render_chat(user_input=None, agent_response=None, is_warning=False):
    if user_input:
        st.markdown(f"""
            <div style='text-align: right; margin-bottom: 10px;'>
                <div style='display: inline-block; background-color: #cce5ff; 
                            color: black; padding: 10px 15px; 
                            border-radius: 15px; max-width: 75%;'>
                    <b>You:</b><br>{user_input}
                </div>
            </div>
        """, unsafe_allow_html=True)

    if is_warning and agent_response:
        st.markdown(f"""
            <div style='text-align: left; margin-bottom: 20px;'>
                <div style='display: inline-block; background-color: #ffcccc; 
                            color: black; padding: 10px 15px; 
                            border-radius: 15px; max-width: 75%;'>
                    <b>⚠️ Travel Assistant Warning:</b><br>{agent_response}
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif agent_response:
        st.markdown(f"""
            <div style='text-align: left; margin-bottom: 20px;'>
                <div style='display: inline-block; background-color: #f1f0f0; 
                            color: black; padding: 10px 15px; 
                            border-radius: 15px; max-width: 75%;'>
                    <b>Travel Assistant 🤖:</b><br>{agent_response}
                </div>
            </div>
        """, unsafe_allow_html=True)


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Agentic Voyage", page_icon="🧳")
st.title("✈ Agentic Voyage...")

query = st.text_input("Ask your Agentic AI Powered travel Agent about your travel Itinerary?")

if query:
    with st.spinner("Planning your trip..."):
        try:
            # Step 1: Extract intent + slots
            intent, new_slots = get_intent_and_slots(query)
            print("Intent and slots extracted:", new_slots)

            # Step 2: Merge into memory
            slots = update_slots(new_slots)
            print("Merged slots memory:", slots)

            # Step 3: Save user bubble
            st.session_state.chat_history.append((query, None, False))

            # Step 4: Apply input guardrails
            guard_check = input_guardrails(slots)
            if guard_check["blocked"]:
                warning_text = f"{guard_check['response']}"
                st.session_state.chat_history.append((None, warning_text, True))
            else:
                # Step 5: Run pipeline
                destination_result = recommend_destinations(slots)
                print("Destination result extracted...")

                itinerary = generate_itinerary(destination_result, slots)
                print("Itinerary result generated...")

                explanation = generate_explanation(itinerary, destination_result)
                print("Explanation generated...")

                # Step 6: Apply output guardrails
                final_output = output_guardrails({
                    "itinerary": itinerary,
                    "explanation": explanation
                })

                formatted_response = format_response(final_output)

                evaluated_response = evaluate_response(formatted_response)

                print("Evaluated response generated...", evaluated_response)

                # Step 7: Save assistant bubble
                st.session_state.chat_history.append((None, formatted_response, False))

        except Exception as e:
            error_msg = f"⚠️ Sorry, an error occurred: {str(e)}"
            st.session_state.chat_history.append((None, error_msg, True))

# -------------------------
# Display conversation
# -------------------------
for user_input, assistant_reply, is_warning in st.session_state.chat_history:
    render_chat(user_input, assistant_reply, is_warning)

# -------------------------
# Styling (auto scroll)
# -------------------------
st.markdown("""
<style>
    .block-container {
        overflow-y: auto;
        height: 80vh;
    }
</style>
""", unsafe_allow_html=True)

print("Loading chunks")
embed_pdf_to_pinecone()