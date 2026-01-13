from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




# API Keysa 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
LLM_MODEL = "llama-3.1-8b-instant"



llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
    temperature=0.4
)

