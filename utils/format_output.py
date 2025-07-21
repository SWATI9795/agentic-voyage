def format_response(response: dict) -> str:
    """
    Format the final response including the itinerary and explanation.

    Parameters:
        response (dict): {
            "itinerary": str or dict,
            "explanation": str or dict or list
        }

    Returns:
        str: Formatted markdown string for display
    """
    formatted = ""

    # Format Itinerary
    itinerary = response.get("itinerary")
    if isinstance(itinerary, dict):
        formatted += "### 🗓️ Your Travel Itinerary\n"
        for day, plan in itinerary.items():
            if isinstance(plan, list):
                plan_text = ", ".join(str(p) for p in plan)
            else:
                plan_text = str(plan)
            formatted += f"**{day}**: {plan_text.strip()}\n\n"
    else:
        formatted += "### 🗓️ Your Travel Itinerary\n"
        formatted += str(itinerary).strip() + "\n\n"

    # Format Explanation
    explanation = response.get("explanation")
    if explanation:
        formatted += "---\n"
        formatted += "### 🤖 Why These Suggestions?\n"
        if isinstance(explanation, dict):
            explanation_text = "\n".join(f"- **{k}**: {v}" for k, v in explanation.items())
        elif isinstance(explanation, list):
            explanation_text = "\n".join(f"- {str(item)}" for item in explanation)
        else:
            explanation_text = str(explanation)
        formatted += explanation_text.strip()

    return formatted
