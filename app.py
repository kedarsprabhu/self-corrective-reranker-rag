import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from agent_lib.graph import build_graph
from ingestion_utils import (
    chunk_and_embed,
    download_file_from_b2,
    extract_text_and_images,
    get_b2_resource,
    upload_file_to_b2,
)
from utils import DatabaseManager
from utils.database import collection
from utils.utils import create_jwt_token, verify_jwt_token

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_manager = DatabaseManager()

# ─── Request / Response Schemas ──────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    query: str
    chat_session: str
    source: List[str]

# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")

    if not db_manager.initialize_pool(min_conn=2, max_conn=20):
        logger.error("Failed to initialize database connection pool")
        raise Exception("Database initialization failed")

    await db_manager.connection_pool.open()
    logger.info("Database pool opened")

    if not await db_manager.create_content_table():
        logger.error("Failed to create content table")
        raise Exception("Table creation failed")

    logger.info("Database initialized successfully")
    yield

    logger.info("Shutting down application...")
    if db_manager.connection_pool:
        await db_manager.connection_pool.close()
        logger.info("Database connection pool closed")
    logger.info("Application shutdown complete")

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Self RAG Service",
    description="RAG-powered document Q&A with streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Root Route ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the chat UI."""
    return FileResponse("static/index.html")

# ─── Auth ────────────────────────────────────────────────────────────────────

@app.post("/token", tags=["auth"])
def token(client_id: str = Body(...), client_secret: str = Body(...)):
    """Get a JWT access token."""
    if (
        client_id == os.getenv("CLIENT_ID", "myclient")
        and client_secret == os.getenv("CLIENT_SECRET", "mysecret")
    ):
        access_token = create_jwt_token(client_id)
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid client credentials")

# ─── System ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# ─── Documents ───────────────────────────────────────────────────────────────

@app.get("/v1/documents", tags=["documents"])
async def list_documents(client: str = Depends(verify_jwt_token)):
    """List all uploaded documents."""
    documents = await db_manager.list_all_files()
    return {"documents": documents}


@app.post("/v1/process-document", response_class=JSONResponse, tags=["documents"])
async def upload_to_b2(
    file: UploadFile = File(...),
    object_name: Optional[str] = Form(None),
    b2_resource=Depends(get_b2_resource),
    extract_images: Optional[bool] = True,
    client: str = Depends(verify_jwt_token),
):
    """Upload a document to B2, download it, extract text, chunk and embed."""
    bucket_name = os.getenv("B2_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="B2_BUCKET_NAME not set in environment")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.flush()
        temp_file_path = temp_file.name

        if not object_name:
            object_name = file.filename

        upload_success = upload_file_to_b2(
            b2_resource=b2_resource,
            local_file_path=temp_file_path,
            bucket_name=bucket_name,
            object_name=object_name,
        )
        logger.info(f"File uploaded to {bucket_name}/{object_name}")
        if not upload_success:
            raise HTTPException(status_code=500, detail="Failed to upload file to B2")

    # Download back for processing
    download_file_path = f"internal_{object_name}"
    download_file = download_file_from_b2(
        b2_resource=b2_resource,
        bucket_name=bucket_name,
        object_name=object_name,
        local_file_path=download_file_path,
    )

    content_id = await db_manager.save_content_db(
        file_name=download_file_path,
        object_key=object_name,
    )
    logger.info(f"Download logged to database with ID: {content_id}")

    if not download_file:
        raise HTTPException(
            status_code=500,
            detail="File uploaded to B2, but failed to download for processing",
        )

    try:
        result = extract_text_and_images(
            downloaded_file_path=download_file_path,
            extract_images=extract_images,
        )
        chunk_and_embed(
            documents=result["text"],
            file_id=content_id,
            database_manager=db_manager,
        )
        logger.info("Document chunked and embedded successfully")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"File uploaded to {bucket_name}/{object_name} and processed",
                "file_path": download_file,
            },
        )
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return JSONResponse(
            status_code=206,
            content={
                "status": "partial_success",
                "message": f"File uploaded but processing failed: {str(e)}",
                "file_path": download_file_path,
            },
        )

# ─── Chat ────────────────────────────────────────────────────────────────────

@app.post("/v1/chat-completion", tags=["chat"])
async def chat_with_context(
    request: ChatCompletionRequest,
    client: str = Depends(verify_jwt_token),
):
    """Stream a chat completion response for the given query and source documents."""
    try:
        file_map = await db_manager.get_file_ids_by_names(request.source)

        if not file_map:
            raise HTTPException(status_code=404, detail="No matching files found")

        missing = set(request.source) - set(file_map.keys())
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Files not found: {', '.join(missing)}",
            )

        file_ids = list(file_map.values())

        groq_api_key = os.environ["GROQ_API_KEY"]
        llm = ChatGroq(
            temperature=0.2,
            api_key=groq_api_key,
            model_name="moonshotai/kimi-k2-instruct-0905",
        )

        langfuse_handler = CallbackHandler(
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host=os.environ.get("LANGFUSE_HOST"),
        )
        try:
            langfuse_handler.auth_check()
        except:
            pass

        graph = build_graph(
            pg_pool=db_manager.connection_pool,
            llm=llm,
            chroma_collection=collection,
        )

        async def generate_stream():
            try:
                inputs = {
                    "query": request.query,
                    "file_ids": file_ids,
                    "chat_session": request.chat_session,
                    "session_id": request.chat_session,
                }

                final_answer = ""
                supporting_facts = []
                confidence_score = None
                sources = request.source

                # State machine for stripping JSON scaffolding from streamed tokens.
                # BUFFERING: accumulate tokens until we find the "answer" value opening quote
                # STREAMING: emit only the answer text to the client
                # DONE: answer value closed, skip remaining JSON tokens
                stream_state = "BUFFERING"
                json_buffer = ""
                escape_next = False

                async for event in graph.astream_events(
                    inputs, 
                    version="v2", 
                    config={"callbacks": [langfuse_handler]}
                ):
                    kind = event["event"]

                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if not content:
                            continue

                        if stream_state == "BUFFERING":
                            json_buffer += content
                            match = re.search(r'"answer"\s*:\s*"', json_buffer)
                            if match:
                                stream_state = "STREAMING"
                                remaining = json_buffer[match.end():]
                                json_buffer = ""
                                answer_chunk = ""
                                for ch in remaining:
                                    if escape_next:
                                        answer_chunk += ch
                                        escape_next = False
                                    elif ch == '\\':
                                        escape_next = True
                                        answer_chunk += ch
                                    elif ch == '"':
                                        stream_state = "DONE"
                                        break
                                    else:
                                        answer_chunk += ch
                                if answer_chunk:
                                    yield f"data: {json.dumps({'event': 'text', 'data': answer_chunk})}\n\n"
                                    final_answer += answer_chunk

                        elif stream_state == "STREAMING":
                            answer_chunk = ""
                            for ch in content:
                                if escape_next:
                                    answer_chunk += ch
                                    escape_next = False
                                elif ch == '\\':
                                    escape_next = True
                                    answer_chunk += ch
                                elif ch == '"':
                                    stream_state = "DONE"
                                    break
                                else:
                                    answer_chunk += ch
                            if answer_chunk:
                                yield f"data: {json.dumps({'event': 'text', 'data': answer_chunk})}\n\n"
                                final_answer += answer_chunk

                        # DONE state: skip remaining JSON tokens silently

                    elif kind == "on_chain_end" and event["name"] == "generate":
                        output = event["data"].get("output")
                        if output and isinstance(output, dict):
                            if "answer" in output:
                                final_answer = output["answer"]
                            if "supporting_facts" in output:
                                supporting_facts = output["supporting_facts"]
                            if "confidence_score" in output:
                                confidence_score = output["confidence_score"]

                final_response = {
                    "event": "final_response",
                    "data": {
                        "answer": final_answer,
                        "sources": sources,
                        "supporting_facts": supporting_facts,
                        "confidence_score": confidence_score,
                    },
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                langfuse_handler.flush()

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, loop="none")
