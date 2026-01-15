from langgraph.graph import StateGraph
from src.langgraphagenticai.state.state import State
from langgraph.graph import START,END
from src.langgraphagenticai.nodes.nodes import BasicChatbotNode


class GraphBuilder:
    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """

        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        # self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        # self.graph_builder.add_edge(START,"chatbot")
        # self.graph_builder.add_edge("chatbot",END)

        self.graph_builder.add_node("parse", self.basic_chatbot_node.parse_document_agent)
        self.graph_builder.add_node("extract", self.basic_chatbot_node.eponym_extraction_agent)
        self.graph_builder.add_node("validate", self.basic_chatbot_node.validation_agent)
        self.graph_builder.add_node("enrich", self.basic_chatbot_node.enrichment_agent)
        self.graph_builder.add_node("song", self.basic_chatbot_node.song_generation_agent)

        self.graph_builder.set_entry_point("parse")
        self.graph_builder.add_edge("parse", "extract")
        self.graph_builder.add_edge("extract", "validate")
        self.graph_builder.add_edge("validate", "enrich")
        self.graph_builder.add_edge("enrich", "song")
        self.graph_builder.add_edge("song", END)


    def setup_graph(self):
        """
        Sets up the graph for the selected use case.
        """
        # if usecase == "Basic Chatbot":
        self.basic_chatbot_build_graph()

        return self.graph_builder.compile()








# from langgraph.graph import StateGraph, END
# from state.schema import MedicalState

# from agents.parser_agent import parse_document_agent
# from agents.extraction_agent import eponym_extraction_agent
# from agents.validation_agent import validation_agent
# from agents.enrichment_agent import enrichment_agent
# from agents.song_agent import song_generation_agent

# def build_graph():
#     graph = StateGraph(MedicalState)

#     self.graph_builder.add_node("parse", parse_document_agent)
#     self.graph_builder.add_node("extract", eponym_extraction_agent)
#     self.graph_builder.add_node("validate", validation_agent)
#     self.graph_builder.add_node("enrich", enrichment_agent)
#     self.graph_builder.add_node("song", song_generation_agent)

#     self.graph_builder.set_entry_point("parse")
#     self.graph_builder.add_edge("parse", "extract")
#     self.graph_builder.add_edge("extract", "validate")
#     self.graph_builder.add_edge("validate", "enrich")
#     self.graph_builder.add_edge("enrich", "song")
#     self.graph_builder.add_edge("song", END)

#     return graph.compile()
