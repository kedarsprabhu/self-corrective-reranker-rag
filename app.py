import asyncio
from contextlib import asynccontextmanager
import json
import logging
import psycopg2
from pydantic import BaseModel
import uvicorn, os
import tempfile
from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Depends
from typing import List, Optional
from fastapi.responses import JSONResponse, StreamingResponse
from utils import DatabaseManager
from pydantic import BaseModel, Field
from ingestion_utils import upload_file_to_b2, get_b2_resource, download_file_from_b2, extract_text_and_images
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ingestion_utils import chunk_and_embed
from utils.utils import create_jwt_token, verify_jwt_token
from agent_lib.graph import build_graph
from utils.database import collection

load_dotenv()
db_manager = DatabaseManager()

class AnswerSchema(BaseModel):
    answer: str = Field(..., description="Final helpful answer to the user's query")
    sources: list[str] = Field(..., description="List of source filenames used to answer")

class ChatCompletionRequest(BaseModel):
    query: str
    chat_session: str
    source: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up application...")
    
    if not db_manager.initialize_pool(min_conn=2, max_conn=20):
        logging.error("Failed to initialize database connection pool")
        raise Exception("Database initialization failed")
    
    if not db_manager.create_content_table():
        logging.error("Failed to create content table")
        raise Exception("Table creation failed")
    
    logging.info("Database initialized successfully")
    
    yield

    logging.info("Shutting down application...")
    if db_manager.connection_pool:
        db_manager.connection_pool.closeall()
        logging.info("Database connection pool closed")
    logging.info("Application shutdown complete")

app = FastAPI(
    title= "Self RAG service",
    description= "API service with multiple endpoints including B2 file upload",
    version= "1.0.0",
    lifespan=lifespan
)

from fastapi import Body

@app.post("/token")
def token(client_id: str = Body(...), client_secret: str = Body(...)):
    # validate client_id and client_secret
    if (
        client_id == os.getenv("CLIENT_ID", "myclient")
        and client_secret == os.getenv("CLIENT_SECRET", "mysecret")
    ):
        token = create_jwt_token(client_id)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid client credentials")


def get_file_ids_by_names(file_names: List[str]):
    sql = """
    SELECT file_name, id
    FROM content
    WHERE file_name = ANY(%s);
    """
    mapping = {}
    with db_manager.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_names,))
            rows = cur.fetchall()
            mapping = {row["file_name"]: row["id"] for row in rows}
    print("map:",mapping)
    return mapping

@app.get("/health",tags=["system"])
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/process-document", response_class=JSONResponse, tags=["storage"])
async def upload_to_b2(
    file: UploadFile = File(...),
    object_name: Optional[str] = Form(None),
    b2_resource = Depends(get_b2_resource),
    extract_images: Optional[bool]=True,
    client: str = Depends(verify_jwt_token)
):
    bucket_name = os.getenv("B2_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="B2_BUCKET_NAME not set in environment")
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
            object_name=object_name
        )
        
        logging.info(f"File uploaded to {bucket_name}/{object_name}")
        if not upload_success:
            raise HTTPException(status_code=500, detail="Failed to upload file to B2")
            
    download_file_path = f"internal_{object_name}"
    download_file = download_file_from_b2(
        b2_resource=b2_resource,
        bucket_name=bucket_name,
        object_name=object_name,
        local_file_path=download_file_path
    )

    content = db_manager.save_content_db(
        file_name=download_file_path,
        object_key=object_name
    )

    logging.info(f"Download logged to database with ID: {content}")

    if not download_file:
        raise HTTPException(status_code=500, detail="File uploaded to B2, but failed to download for processing")

    try:
        result=extract_text_and_images(downloaded_file_path=download_file_path, extract_images=extract_images)
        print("result:", result)
        
        chunk_and_embed(documents=result["text"],file_id=content, database_manager=db_manager)
        logging.info("chunked and embedded")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"File uploaded to {bucket_name}/{object_name} and processed",
                "file_path": download_file
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=206,
            content={
                "status": "partial_success",
                "message": f"File uploaded but processing failed: {str(e)}",
                "file_path": download_file_path
            }
        )

@app.post("/v1/chat-completion", response_class=JSONResponse, tags=["chat"])
async def chat_with_context(
    request: ChatCompletionRequest,
    client: str = Depends(verify_jwt_token)
):
    try:
        # 1. Map file_name → file_id
        file_map = get_file_ids_by_names(request.source)

        if not file_map:
            raise HTTPException(status_code=404, detail="No matching files found")

        # Ensure all requested sources exist
        missing = set(request.source) - set(file_map.keys())
        if missing:
            raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(missing)}")

        file_ids = list(file_map.values())

        # 2. Compile Graph
        # We need to initialize the graph with resources
        # Ideally, we should cache this or make it efficient
        groq_api_key = os.environ['GROQ_API_KEY']
        llm = ChatGroq(
            temperature=0.2,
            api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )
        
        graph = build_graph(
            pg_pool=db_manager.connection_pool,
            llm=llm,
            chroma_collection=collection
        )

        async def generate_stream():
            try:
                inputs = {
                    "query": request.query,
                    "file_ids": file_ids,
                    "chat_session": request.chat_session,
                    "session_id": request.chat_session # using chat_session as session_id
                }
                
                async for event in graph.astream(inputs):
                     for key, value in event.items():
                         # We are interested in the final answer from 'generate' node
                         # OR streaming updates if we want to show progress
                         if key == "generate":
                             api_response = {
                                 "answer": value.get("answer"),
                                 "is_relevant": value.get("is_relevant"),
                                 "sources": request.source # Passing back original source list or we could extract from docs
                             }
                             yield f"data: {json.dumps(api_response)}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                yield json.dumps({"error": str(e)})

        # 5️⃣ Return a streaming JSON response
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

