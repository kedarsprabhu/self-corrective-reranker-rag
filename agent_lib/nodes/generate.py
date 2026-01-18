import os
from typing import List, Optional

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langfuse.callback import CallbackHandler
from ..state import GraphState


class AnswerSchema(BaseModel):
    summary: str = Field(
        ...,
        description="Detailed and comprehensive answer to the query"
    )
    supporting_facts: List[str] = Field(
        ...,
        description="Facts from the context supporting the summary"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )


class Generate:
    name = "generate"

    def __init__(self):
        # Keep init minimal
        self.model_name = "llama3-8b-8192"
        self.temperature = 0.2

    async def __call__(self, state: GraphState) -> GraphState:
        """
        Generate a structured answer using reranked documents.
        """



        llm = ChatGroq(
            temperature=self.temperature,
            api_key=os.environ["GROQ_API_KEY"],
            model_name=self.model_name,
        )

        langfuse_handler = CallbackHandler(
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host=os.environ.get("LANGFUSE_HOST"),
        )

        try:
            langfuse_handler.auth_check()
        except Exception:
            pass

        # ---------------------------
        # Parser
        # ---------------------------

        parser = PydanticOutputParser(
            pydantic_object=AnswerSchema
        )

        # ---------------------------
        # Prompt
        # ---------------------------

        prompt = PromptTemplate.from_template(
        """
        Use the context below to answer the question.
        If the context does not contain relevant information, say you do not know.

        Context:
        {context}

        Question:
        {query}

        Your response MUST be a valid JSON object with the following keys:
        - summary
        - supporting_facts
        - confidence_score

        - confidence_score should be LOW (close to 0)
        if the context does NOT adequately answer the question
        - confidence_score should be HIGH (close to 1)
        only if the answer clearly addresses the question

        {format_instructions}

        IMPORTANT:
        - Do NOT add any text outside the JSON
        - Do NOT add explanations or prefixes
        """
        )


        chain = prompt | llm | parser


        context = "\n\n".join(state.get("reranked_documents", []))

        try:
            result: AnswerSchema = await chain.ainvoke(
                {
                    "context": context,
                    "query": state["query"],
                    "format_instructions": parser.get_format_instructions(),
                },
                config={
                    "callbacks": [langfuse_handler]
                },
            )
            

            return {
                **state,
                "answer": result.summary,
                "is_relevant": result.confidence_score >= 0.6
            }

        except Exception as e:
            print("Generation error:", e)

            return {
                **state,
                "answer": (
                    "I encountered an issue while generating the answer. "
                    "Please try rephrasing the question or providing more context."
                ),
            }
