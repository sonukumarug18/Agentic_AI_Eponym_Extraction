"""Graph builder for LangGraph RAG workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:
    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        builder = StateGraph(RAGState)

        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("generate", self.nodes.generate_answer)
        builder.add_node("rewrite", self.nodes.rewrite)

        builder.set_entry_point("retrieve")

        builder.add_conditional_edges(
            "retrieve",
            self.nodes.grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )

        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> RAGState:
        if self.graph is None:
            self.build()

        initial_state = RAGState(
            question=question,
            retrieved_docs=[],
            answer=""   # ✅ NEVER None
        )

        return self.graph.invoke(initial_state)
