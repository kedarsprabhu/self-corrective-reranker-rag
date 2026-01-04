from langchain_groq import ChatGroq
from langfuse.callback import CallbackHandler
import os

groq_api_key = os.environ['GROQ_API_KEY']

def __get_llm(model_name:str):
    llm = ChatGroq(
        temperature=0.2,
        api_key=groq_api_key,
        model_name=model_name    #"llama3-8b-8192"
    )
    return llm


from typing import List, Dict, Any

class ChromaRetriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(
        self,
        query: str,
        file_ids: List[str],
        top_k: int = 12
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"file_id": {"$in": file_ids}}
        )

        docs = []
        if not results or not results.get("ids"):
            return docs

        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return docs
