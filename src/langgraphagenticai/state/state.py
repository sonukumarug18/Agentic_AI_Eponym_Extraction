from typing_extensions import TypedDict,List,Dict,Optional
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """
    Represent the structure of the state used in graph
    """
    file_path: str
    raw_text: str
    candidate_eponyms: List[Dict]
    validated_eponyms: List[Dict]
    enriched_eponyms: List[Dict]
    medical_song: str
    error: Optional[str]

