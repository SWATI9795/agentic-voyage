import re

def clean_text(text: str) -> str:
    """Collapse weird newlines/spaces into one space."""
    return re.sub(r"\s+", " ", text).strip()

def format_response(response: dict) -> str:
    """
    Nicely format itinerary + explanation into Markdown.
    """

    formatted = "## 🧳 Your Personalized Travel Plan\n"

    # ---------- ITINERARY ----------
    itinerary = response.get("itinerary")
    if isinstance(itinerary, dict):
        for i, (day, details) in enumerate(itinerary.items(), start=1):
            formatted += f"\n### Day {i}\n"

            # Activities
            formatted += "🎯 **Activities & Places to Visit:**\n"
            activities = details.get("activities", [])
            if activities:
                for act in activities:
                    formatted += f"- 🏞️ {act}\n"
            else:
                formatted += "- No activities found\n"

            # Stay
            stay = details.get("stay", "No hotel info")
            formatted += f"\n🏨 **Suggested Stay:** {stay}\n"

            # Small trick: make hotel names clickable if destination is mentioned
            if "Udaipur" in stay:
                formatted += "🔗 [Browse hotels in Udaipur](https://www.booking.com/city/in/udaipur.html)\n"
            elif "Shimla" in stay:
                formatted += "🔗 [Browse hotels in Shimla](https://www.booking.com/city/in/shimla.html)\n"
            elif "Chandigarh" in stay:
                formatted += "🔗 [Browse hotels in Chandigarh](https://www.booking.com/city/in/chandigarh.html)\n"

            # Description
            description = details.get("description", "")
            if description:
                formatted += f"\n📖 **About this Day:** {description}\n"

    else:
        formatted += f"\n{str(itinerary)}\n"

    # ---------- EXPLANATION ----------
    explanation = response.get("explanation")
    if explanation:
        formatted += "\n---\n"
        formatted += "## 🤖 Why These Suggestions?\n"
        formatted += explanation.strip()

    return formatted