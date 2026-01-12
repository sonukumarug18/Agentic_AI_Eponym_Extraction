"""LangGraph nodes for RAG workflow"""

from typing import List, Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from src.state.rag_state import RAGState

# --------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# --------------------------------------------------


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        #  Plain text LLMs 
        self.grader_llm = ChatGroq(model="llama-3.1-8b-instant")
        self.rewrite_llm = ChatGroq(model="llama-3.1-8b-instant")

    # --------------------------------------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        print("---RETRIEVE---")

        docs: List[Document] = self.retriever.invoke(state.question)

        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            answer=""
        )

    # --------------------------------------------------
    def grade_documents(self, state: RAGState) -> Literal["generate", "rewrite"]:
        print("---GRADE DOCUMENTS---")

        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = PromptTemplate(
            template="""
You are a strict grader.

Question:
{question}

Context:
{context}

Does the context contain information that can answer the question?

Reply with ONLY one word:
yes or no
""",
            input_variables=["context", "question"],
        )

        response = self.grader_llm.invoke(
            prompt.format(
                question=state.question,
                context=context
            )
        )

        decision = response.content.strip().lower()

        print("Grader output:", decision)

        if "yes" in decision:
            return "generate"
        else:
            return "rewrite"

    # --------------------------------------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        print("---GENERATE ANSWER---")

        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{state.question}
"""

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )

    # --------------------------------------------------
    def rewrite(self, state: RAGState) -> RAGState:
        print("---REWRITE QUERY---")

        prompt = f"""
Rewrite the question to improve document retrieval.

Original Question:
{state.question}

Rewritten Question:
"""

        response = self.rewrite_llm.invoke(prompt)

        return RAGState(
            question=response.content.strip(),
            retrieved_docs=[],
            answer=""
        )
