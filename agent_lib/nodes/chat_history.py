from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from state import GraphState
import os


from typing import List

POSTGRES_URL = os.environ["POSTGRES_URL"]


class SetChatHistory:
    name = "set_chat_history"

    def __init__(self, async_pool):
        self.async_pool = async_pool

    async def __call__(self, state: GraphState) -> GraphState:
        history = PostgresChatMessageHistory(
            table_name="chat_history",
            session_id=state["session_id"],
            async_connection_pool=self.async_pool
        )

        messages: List[BaseMessage] = await history.aget_messages()

        return {
            **state,
            "chat_history": messages
        }



class StoreChatHistory:
    name = "store_chat_history"

    def __init__(self, async_pool):
        self.async_pool = async_pool

    async def __call__(self, state: GraphState) -> GraphState:
        history = PostgresChatMessageHistory(
            table_name="chat_history",
            session_id=state["session_id"],
            async_connection_pool=self.async_pool
        )

        await history.aadd_message(
            HumanMessage(content=state["query"])
        )

        await history.aadd_message(
            AIMessage(content=state["final_answer"])
        )

        return state

