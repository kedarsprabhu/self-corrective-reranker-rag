from utils.utils import BM25Reranker
from ..state import GraphState
from ..utils import ChromaRetriever

class Retrieve:
    name = "retrieve"

    def __init__(
        self,
        top_k_retrieve: int = 15,
        top_k_rerank: int = 8,
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.reranker = BM25Reranker()

    async def __call__(self, state: GraphState) -> GraphState:
        try:
            # 1️⃣ Retrieve from Chroma
            raw_docs = ChromaRetriever.retrieve(
                query=state["query"],
                file_ids=state.get("file_ids", []),
                top_k=self.top_k_retrieve
            )

            texts = [d["text"] for d in raw_docs] if raw_docs else []

            # 2️⃣ BM25 rerank
            reranked_docs = self.reranker.rerank(
                query=state["query"],
                documents=texts,
                top_k=self.top_k_rerank
            )

            return {
                **state,
                "documents": texts,
                "reranked_documents": reranked_docs
            }

        except Exception as e:
            print("Retrieve + BM25 error:", e)
            return {
                **state,
                "documents": [],
                "reranked_documents": []
            }
