import os
import tempfile
import streamlit as st
from graph.workflow import build_graph

st.set_page_config(page_title="Medical Eponym AI", layout="wide")
st.title("🧠 Medical Eponym Analyzer & Song Generator")

uploaded_file = st.file_uploader(
    "Upload Medical Document (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    # ✅ Preserve original extension
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if st.button("🚀 Process Document"):
        with st.spinner("Analyzing document with Agentic AI..."):
            initial_state = {
                "file_path": file_path,
                "raw_text": "",
                "candidate_eponyms": [],
                "validated_eponyms": [],
                "enriched_eponyms": [],
                "medical_song": "",
                "error": None
            }

            app = build_graph()
            final_state = app.invoke(initial_state)

        if final_state["error"]:
            st.error(final_state["error"])
        else:
            st.success("Processing completed!")

            st.subheader("✅ Enriched Medical Eponyms")
            for e in final_state["enriched_eponyms"]:
                st.markdown(
                    f"""
**{e['name']}**  
- Named After: {e['named_after']}  
- System: {e['system']}  
- Description: {e['description']}
"""
                )

            st.subheader("🎵 Generated Medical Song")
            st.text_area("Song", final_state["medical_song"], height=300)
