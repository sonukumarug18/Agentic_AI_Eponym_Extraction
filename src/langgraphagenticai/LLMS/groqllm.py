import os
import streamlit as st
from langchain_groq import ChatGroq


class GroqLLM:
    def __init__(self,GROQ_API_KEY,LLM_MODEL):
        self.GROQ_API_KEY = GROQ_API_KEY
        self.LLM_MODEL=LLM_MODEL

    def get_llm_model(self):
        try:
            # groq_api_key=self.GROQ_API_KEY
            # selected_groq_model=self.LLM_MODEL
            # if groq_api_key=='' and os.environ["GROQ_API_KEY"] =='':
            #     st.error("Please Enter the Groq API KEY")

            llm=ChatGroq(api_key=self.GROQ_API_KEY,model=self.LLM_MODEL)

        except Exception as e:
            raise ValueError(f"Error Ocuured With Exception : {e}")
        return llm


