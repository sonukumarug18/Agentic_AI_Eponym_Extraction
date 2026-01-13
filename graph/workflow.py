from langgraph.graph import StateGraph, END
from state.schema import MedicalState

from agents.parser_agent import parse_document_agent
from agents.extraction_agent import eponym_extraction_agent
from agents.validation_agent import validation_agent
from agents.enrichment_agent import enrichment_agent
from agents.song_agent import song_generation_agent

def build_graph():
    graph = StateGraph(MedicalState)

    graph.add_node("parse", parse_document_agent)
    graph.add_node("extract", eponym_extraction_agent)
    graph.add_node("validate", validation_agent)
    graph.add_node("enrich", enrichment_agent)
    graph.add_node("song", song_generation_agent)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "enrich")
    graph.add_edge("enrich", "song")
    graph.add_edge("song", END)

    return graph.compile()
