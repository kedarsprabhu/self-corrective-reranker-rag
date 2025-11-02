import os
from fastapi import HTTPException
import jwt, datetime
from fastapi import Header, HTTPException, status

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