from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import SetChatHistory, StoreChatHistory, Generate, Retrieve, Planner
from .edges import should_retry

def build_graph(pg_pool, llm, chroma_collection):
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node(
        "set_chat_history",
        SetChatHistory(pg_pool),
    )
    workflow.add_node(
        "planner",
        Planner(llm),
    )
    workflow.add_node(
        "retrieve",
        Retrieve(chroma_collection=chroma_collection),   # Chroma + BM25 inside
    )
    workflow.add_node(
        "generate",
        Generate(),   # sets answer + is_relevant
    )
    workflow.add_node(
        "store_chat_history",
        StoreChatHistory(pg_pool),
    )

    # Entry
    workflow.set_entry_point("set_chat_history")

    # Main path
    workflow.add_edge("set_chat_history", "planner")
    workflow.add_edge("planner", "retrieve")
    workflow.add_edge("retrieve", "generate")

    # Conditional retry
    workflow.add_conditional_edges(
        "generate",
        should_retry,
        path_map={
            "retry": "planner",  
            "__end__": "store_chat_history",
        },
    )

    # Finish
    workflow.add_edge("store_chat_history", END)

    return workflow.compile()