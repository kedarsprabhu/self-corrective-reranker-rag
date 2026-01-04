from state import GraphState
from utils import ChromaRetriever


class Retrieve:
    name = "retrieve"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    async def __call__(self, state: GraphState) -> GraphState:
        try:
            docs = ChromaRetriever.retrieve(
                query=state["query"],
                file_ids=state.get("file_ids", []),
                top_k=self.top_k
            )

            documents = (
                [d["text"] for d in docs]
                if docs
                else ["No relevant information found in the knowledge base."]
            )

            return {
                **state,
                "documents": documents
            }

        except Exception as e:
            print("Chroma retrieval error:", e)
            return {
                **state,
                "documents": ["Unable to retrieve information at this time."]
            }