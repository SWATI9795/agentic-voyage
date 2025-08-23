def apply_guardrails(itinerary, explanation, slots):
    sensitive_keywords = ["visa", "passport", "credit card", "insurance", "legal"]
    restricted_places = ["north korea", "gaza", "syria"]

    query_text = " ".join(str(v).lower() for v in slots.values() if isinstance(v, str))

    if any(k in query_text for k in sensitive_keywords):
        return "⚠️ I'm not authorized to provide sensitive or financial advice."

    if any(loc in query_text for loc in restricted_places):
        return "⚠️ Travel to this location is restricted. Please choose another destination."

    return {
        "itinerary": itinerary,
        "explanation": explanation
    }

def input_guardrails(slots: dict):
    """
    Check for unsafe or restricted destinations in user input (slots).
    """
    restricted_places = ["syria", "ukraine", "north korea", "gaza", "afghanistan"]

    # Check destination slot
    destination = str(slots.get("destination", "")).lower()
    if any(loc in destination for loc in restricted_places):
        return {
            "blocked": True,
            "response": {
                "itinerary": "⚠️ Travel Advisory",
                "explanation": (
                    f"The requested destination **{slots.get('destination', '')}** "
                    "is currently considered unsafe for travel due to conflicts or security concerns. "
                    "I cannot provide an itinerary for this location. "
                    "Please consider choosing another destination."
                )
            }
        }
    return {"blocked": False}

def output_guardrails(response: dict):
    """
    Validate final model output to block sensitive/unsafe topics.
    """
    sensitive_keywords = [
        "visa", "passport", "credit card", "insurance",
        "loan", "bank account", "money transfer"
    ]

    # Flatten itinerary + explanation text for scanning
    combined_text = f"{response.get('itinerary', '')} {response.get('explanation', '')}".lower()

    if any(k in combined_text for k in sensitive_keywords):
        return {
            "itinerary": "⚠️ Restricted Information",
            "explanation": (
                "Some parts of the generated response included sensitive or financial guidance "
                "which I am not authorized to provide. "
                "Please focus on destinations, activities, and travel experiences instead."
            )
        }

    return response