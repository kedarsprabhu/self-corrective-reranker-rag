import os
from fastapi import HTTPException
import jwt, datetime

JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")   # put in .env
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60   # token valid for 1 hour

def create_jwt_token(username: str):
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]   # return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
