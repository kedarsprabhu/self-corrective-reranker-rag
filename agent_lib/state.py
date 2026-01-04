from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    query: str
    documents: List[str]
    reranked_documents: List[str]
    answer: Optional[str]
    is_relevant: Optional[bool]
    retry_count: int
    final_answer: Optional[str]