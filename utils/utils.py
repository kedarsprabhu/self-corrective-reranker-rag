import os
from fastapi import HTTPException
import jwt, datetime
from fastapi import Header, HTTPException, status

import re
from rank_bm25 import BM25Okapi
from typing import List

JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")   # put in .env
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60   # token valid for 1 hour

def create_jwt_token(username: str):
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(authorization: str = Header(...)):
    """
    Verifies JWT token from Authorization header (Bearer <token>).
    Returns username if valid.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'",
        )

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _tokenize(text: str) -> List[str]:
    """
    Simple, robust tokenizer for BM25.
    Lowercases, removes punctuation.
    """
    return re.findall(r"\w+", text.lower())


class BM25Reranker:
    def __init__(self):
        pass

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int
    ) -> List[str]:
        if not documents:
            return []

        tokenized_docs = [_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]
