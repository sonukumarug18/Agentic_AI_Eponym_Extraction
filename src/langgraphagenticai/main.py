import streamlit as st
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder

import tempfile
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# API Keysa 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
LLM_MODEL = "llama-3.1-8b-instant"



def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.

    """

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



                ## Configure The LLM's
                obj_llm_config=GroqLLM(GROQ_API_KEY,LLM_MODEL)
                model=obj_llm_config.get_llm_model()

                if not model:
                    st.error("Error: LLM model could not be initialized")
                    return
                

                app=GraphBuilder(model)
                graph=app.setup_graph()
                final_state = graph.invoke(initial_state)

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



