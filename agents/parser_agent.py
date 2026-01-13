import pdfplumber
from docx import Document
import os

def parse_document_agent(state):
    try:
        ext = os.path.splitext(state["file_path"])[1].lower()

        if ext == ".pdf":
            text = ""
            with pdfplumber.open(state["file_path"]) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

        elif ext == ".docx":
            doc = Document(state["file_path"])
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext == ".txt":
            with open(state["file_path"], "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format")

        state["raw_text"] = text
        return state

    except Exception as e:
        state["error"] = f"Parsing failed: {e}"
        return state
