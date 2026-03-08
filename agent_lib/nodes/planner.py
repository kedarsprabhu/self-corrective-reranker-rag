from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..state import GraphState

from langchain_core.messages import SystemMessage, HumanMessage
from ..state import GraphState

class Planner:
    name = "planner"

    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, state: GraphState, config: dict = None) -> GraphState:
        # Build messages manually
        messages = [
            SystemMessage(content=(
                "You are a query planner for a RAG system. "
                "Given the chat history and the latest user question, "
                "rewrite the question into a clear, standalone query "
                "optimized for semantic document retrieval. "
                "Do NOT answer the question. "
                "Return ONLY the rewritten query into a meaningful query based on chat history and current user question."
            )),
            *state.get("chat_history", []), 
            HumanMessage(content=state["query"])
        ]
        
        response = await self.llm.ainvoke(messages, config=config)
        planned_query = response.content.strip()

        # Safety fallback
        if not planned_query:
            planned_query = state["query"]

        return {
            **state,
            "query": planned_query
        }