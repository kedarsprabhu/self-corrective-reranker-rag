import psycopg_pool
from contextlib import asynccontextmanager
from fastapi import FastAPI
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.async_pool = psycopg_pool.AsyncConnectionPool(
        conninfo=os.environ["POSTGRES_URL"],
        min_size=1,
        max_size=10,
        timeout=30
    )
    yield

    await app.state.async_pool.close()


app = FastAPI(lifespan=lifespan)
