import json
from config.settings import llm


def validation_agent(state):
    validated = []

    try:
        for item in state["candidate_eponyms"]:
            prompt = f"""
Validate medical eponym.

Eponym: {item["eponym"]}
Context: {item["context_sentence"]}

Return STRICT JSON only:
{{
  "valid": true,
  "category": "Neurological disease",
  "reason": "Recognized clinical eponym"
}}
"""

            response = llm.invoke(prompt)

            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                continue

            if result.get("valid"):
                validated.append({
                    "name": item["eponym"],
                    "category": result.get("category", ""),
                    "reason": result.get("reason", "")
                })

        state["validated_eponyms"] = validated
        return state

    except Exception as e:
        state["error"] = f"Validation failed: {str(e)}"
        return state
