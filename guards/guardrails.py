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
