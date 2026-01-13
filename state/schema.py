from typing import TypedDict, List, Dict, Optional

class MedicalState(TypedDict):
    file_path: str
    raw_text: str
    candidate_eponyms: List[Dict]
    validated_eponyms: List[Dict]
    enriched_eponyms: List[Dict]
    medical_song: str
    error: Optional[str]
