from langchain.prompts import ChatPromptTemplate

from state import GraphState

class Rerank:
    name = "rerank"

    def __init__(self, llm, top_k: int = 4):
        self.top_k = top_k

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are ranking document chunks for relevance to a query. "
                "Select the most relevant chunks and return them verbatim. "
                "Do NOT rewrite or summarize."
            ),
            ("human", "Query:\n{query}\n\nChunks:\n{chunks}")
        ])

        self.chain = self.prompt | llm

    async def __call__(self, state: GraphState) -> GraphState:
        chunks = "\n\n".join(
            f"[{i}] {doc}" for i, doc in enumerate(state["documents"])
        )

        response = await self.chain.ainvoke({
            "query": state["query"],
            "chunks": chunks
        })

        text = response.content.strip()

        # Fallback: keep first K documents
        reranked = state["documents"][: self.top_k]

        if text:
            # very light heuristic split
            reranked = text.split("\n\n")[: self.top_k]

        return {
            **state,
            "reranked_documents": reranked
        }
