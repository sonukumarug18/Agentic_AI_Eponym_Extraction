from src.langgraphagenticai.state.state import State

import pdfplumber
from docx import Document
import os
import json

from src.langgraphagenticai.utils.chunking import sentence_aware_chunking


class BasicChatbotNode:
    """
    Basic  logic implementation for
    document parsing, eponym extraction,
    validation, enrichment, and song generation.
    """

    def __init__(self, model):
        self.llm = model

    # -------------------- DOCUMENT PARSER --------------------
    def parse_document_agent(self, state: State) -> dict:
        try:
            file_path = state.get("file_path")
            if not file_path:
                raise ValueError("file_path missing in state")

            ext = os.path.splitext(file_path)[1].lower()
            text = ""

            if ext == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

            elif ext == ".docx":
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs if p.text)

            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            else:
                raise ValueError("Unsupported file format")

            state["raw_text"] = text
            return state

        except Exception as e:
            state["error"] = f"Parsing failed: {str(e)}"
            return state

    # -------------------- EPONYM EXTRACTION --------------------
    def eponym_extraction_agent(self, state: State) -> dict:
        try:
            raw_text = state.get("raw_text", "")
            if not raw_text:
                state["candidate_eponyms"] = []
                return state

            chunks = sentence_aware_chunking(raw_text)
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

                response = self.llm.invoke(prompt)

                try:
                    parsed = json.loads(response.content.strip())
                    if isinstance(parsed, list):
                        all_eponyms.extend(parsed)
                except json.JSONDecodeError:
                    continue  # skip bad chunk safely

            state["candidate_eponyms"] = all_eponyms
            return state

        except Exception as e:
            state["error"] = f"Eponym extraction failed: {str(e)}"
            return state

    # -------------------- VALIDATION AGENT --------------------
    def validation_agent(self, state: State) -> dict:
        validated = []

        try:
            for item in state.get("candidate_eponyms", []):
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

                response = self.llm.invoke(prompt)

                try:
                    result = json.loads(response.content.strip())
                except json.JSONDecodeError:
                    continue

                if result.get("valid") is True:
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

    # -------------------- ENRICHMENT AGENT --------------------
    def enrichment_agent(self, state: State) -> dict:
        enriched = []

        try:
            for item in state.get("validated_eponyms", []):
                prompt = f"""
Provide medical enrichment for {item["name"]}.

Return STRICT JSON only:
{{
  "named_after": "",
  "affected_system": "",
  "clinical_description": ""
}}
"""

                response = self.llm.invoke(prompt)

                try:
                    result = json.loads(response.content.strip())
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

    # -------------------- SONG GENERATION --------------------
    def song_generation_agent(self, state: State) -> dict:
        try:
            eponyms = state.get("enriched_eponyms", [])

            if not eponyms:
                state["medical_song"] = "No valid medical eponyms found."
                return state

            eponym_text = "\n".join(
                f"- {e['name']}: {e['description']}"
                for e in eponyms
            )

            prompt = f"""
Write a creative, medically accurate song using these eponyms.
Include verses and a chorus.

{eponym_text}
"""

            response = self.llm.invoke(prompt)
            state["medical_song"] = response.content
            return state

        except Exception as e:
            state["error"] = f"Song generation failed: {str(e)}"
            return state
