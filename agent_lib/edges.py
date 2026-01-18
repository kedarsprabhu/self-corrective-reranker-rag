from .state import GraphState

def should_retry(state: GraphState) -> str:
    """
    Retry only if:
    - answer is not relevant
    - retry_count < 1
    """
    if not state.get("is_relevant") and state.get("retry_count", 0) < 1:
        return "rephrase"
    return "__end__"