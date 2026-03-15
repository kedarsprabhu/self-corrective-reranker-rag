import os
from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler
from ..state import GraphState

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import json

class AnswerSchema(BaseModel):
    answer: str = Field(..., description="Detailed answer")
    supporting_facts: List[str] = Field(..., description="Supporting facts")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class Generate:
    name = "generate"

    def __init__(self):
        self.model_name = "llama-3.1-8b-instant"
        self.temperature = 0.2

    async def __call__(self, state: GraphState, config: Optional[dict] = None) -> GraphState:
        """Generate a structured answer using reranked documents."""
        
        llm = ChatGroq(
            temperature=self.temperature,
            api_key=os.environ["GROQ_API_KEY"],
            model_name=self.model_name,
        )

        # Build context
        context = "\n\n".join(state.get("reranked_documents", []))

        # Build messages manually
        messages = [
            SystemMessage(content="""Use the context below to answer the question.
If the context does not contain relevant information, say you do not know.

Your response MUST be a valid JSON object with these keys:
- answer: Detailed answer
- supporting_facts: List of facts from context
- confidence_score: 0.0 to 1.0 (LOW if context doesn't answer, HIGH if it does)

IMPORTANT: Return ONLY the JSON, no explanations or markdown."""),
            HumanMessage(content=f"""Context:
{context}

Question:
{state["query"]}""")
        ]

        try:
            # Stream-compatible chain
            chain = llm | StrOutputParser()
            
            # Invoke with config for streaming support
            result_text = await chain.ainvoke(messages, config=config)
            
            # Parse JSON manually
            cleaned_text = result_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            parsed_result = json.loads(cleaned_text.strip())
            
            return {
                **state,
                "answer": parsed_result.get("answer", result_text),
                "supporting_facts": parsed_result.get("supporting_facts", []),
                "confidence_score": parsed_result.get("confidence_score"),
                "is_relevant": parsed_result.get("confidence_score", 0.0) >= 0.6
            }
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                **state,
                "answer": result_text,
                "is_relevant": False 
            }
        except Exception as e:
            print("Generation error:", e)
            return {
                **state,
                "answer": "I encountered an issue while generating the answer. Please try rephrasing.",
            }