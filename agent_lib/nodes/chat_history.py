from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from ..state import GraphState
import os
from typing import List

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")


class SetChatHistory:
    name = "set_chat_history"

    def __init__(self, async_pool):
        self.async_pool = async_pool

    async def __call__(self, state: GraphState) -> GraphState:
        async with self.async_pool.connection() as conn:
            history = PostgresChatMessageHistory(
                "chat_history",
                state["session_id"],
                async_connection=conn,
            )
            
            messages: List[BaseMessage] = await history.aget_messages()
            
            # Keep only last 6 conversations (12 messages)
            recent_messages = messages[-12:] if len(messages) > 12 else messages

        return {
            **state,
            "chat_history": recent_messages  # Only recent messages
        }


class StoreChatHistory:
    name = "store_chat_history"

    def __init__(self, async_pool):
        self.async_pool = async_pool

    async def __call__(self, state: GraphState) -> GraphState:
        async with self.async_pool.connection() as conn:
            history = PostgresChatMessageHistory(
                "chat_history",
                state["session_id"],
                async_connection=conn,
            )
            
            # Store current exchange (always stores in DB)
            await history.aadd_messages([
                HumanMessage(content=state["query"]),
                AIMessage(content=state["answer"])
            ])

        return state


class StoreChatHistory:
    name = "store_chat_history"

    def __init__(self, async_pool):
        self.async_pool = async_pool

    async def __call__(self, state: GraphState) -> GraphState:
        async with self.async_pool.connection() as conn:
            history = PostgresChatMessageHistory(
                "chat_history",
                state["session_id"],
                async_connection=conn,
            )
            
            await history.aadd_messages([
                HumanMessage(content=state["query"]),
                AIMessage(content=state["answer"])
            ])

        return state