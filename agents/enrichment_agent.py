import json
from config.settings import llm


def enrichment_agent(state):
    enriched = []

    try:
        for item in state["validated_eponyms"]:
            prompt = f"""
Provide medical enrichment for {item["name"]}.

Return STRICT JSON only:
{{
  "named_after": "",
  "affected_system": "",
  "clinical_description": ""
}}
"""

            response = llm.invoke(prompt)

            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                continue

            enriched.append({
                "name": item["name"],
                "category": item["category"],
                "named_after": result.get("named_after", ""),
                "system": result.get("affected_system", ""),
                "description": result.get("clinical_description", "")
            })

        state["enriched_eponyms"] = enriched
        return state

    except Exception as e:
        state["error"] = f"Enrichment failed: {str(e)}"
        return state
