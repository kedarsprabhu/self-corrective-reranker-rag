from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import GraphState

class Planner:
    name = "planner"

    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query planner for a RAG system. "
                "Given the chat history and the latest user question, "
                "rewrite the question into a clear, standalone query "
                "optimized for semantic document retrieval. "
                "Do NOT answer the question. "
                "Return ONLY the rewritten query."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.llm

    async def __call__(self, state: GraphState) -> GraphState:
        response = await self.chain.ainvoke({
            "query": state["query"],
            "chat_history": state.get("chat_history", [])
        })

        planned_query = response.content.strip()

        # Safety fallback
        if not planned_query:
            planned_query = state["query"]

        return {
            **state,
            "query": planned_query
        }
