import json
from config.settings import llm
from utils.chunking import sentence_aware_chunking


def eponym_extraction_agent(state):
    try:
        chunks = sentence_aware_chunking(state["raw_text"])
        all_eponyms = []

        for chunk in chunks:
            prompt = f"""
You are a medical expert.

Extract ONLY true medical eponyms (diseases, syndromes, signs named after people).
Ignore historical mentions or non-medical names.

Return STRICT JSON only. No explanations. No markdown.

Format:
[
  {{
    "eponym": "Alzheimer's disease",
    "context_sentence": "The patient was diagnosed with Alzheimer's disease."
  }}
]

TEXT:
{chunk["text"]}
"""

            response = llm.invoke(prompt)

            # ✅ SAFE JSON PARSING
            try:
                parsed = json.loads(response.content)
                if isinstance(parsed, list):
                    all_eponyms.extend(parsed)
            except json.JSONDecodeError:
                # Skip bad chunk instead of crashing
                continue

        state["candidate_eponyms"] = all_eponyms
        return state

    except Exception as e:
        state["error"] = f"Eponym extraction failed: {str(e)}"
        return state
