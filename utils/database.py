import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
import psycopg
from psycopg_pool import AsyncConnectionPool
import logging
import os
import uuid
import json
from typing import List
from dotenv import load_dotenv
import chromadb

# Initialize Chroma client (persistent)
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("document_chunks")

load_dotenv(".env")

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@localhost/{db_name}"


class __DatabaseManager:
    def __init__(self):
        """
        Initialize database manager with configuration
        """
        self.db_config = {
            "host": "localhost",
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": 5432
        }
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)

    def initialize_pool(self, min_conn: int = 2, max_conn: int = 20) -> bool:
        """Initialize the async database connection pool"""
        try:
            # Build connection string
            conninfo = (
                f"host={self.db_config['host']} "
                f"port={self.db_config['port']} "
                f"dbname={self.db_config['dbname']} "
                f"user={self.db_config['user']} "
                f"password={self.db_config['password']}"
            )
            
            self.connection_pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=min_conn,
                max_size=max_conn,
                open=False,
            )
            self.logger.info("Async database connection pool created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating connection pool: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        if not self.connection_pool:
            raise Exception("Connection pool not initialized")
        
        async with self.connection_pool.connection() as conn:
            try:
                yield conn
            except Exception as e:
                await conn.rollback()
                self.logger.error(f"Database connection error: {e}")
                raise

    async def create_content_table(self) -> bool:
        """Create the content table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS content (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            object_key TEXT NOT NULL,
            downloaded_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_content_file_name ON content(file_name);
        CREATE INDEX IF NOT EXISTS idx_content_downloaded_on ON content(downloaded_on);
        CREATE INDEX IF NOT EXISTS idx_content_object_key ON content(object_key);
        
        -- Create document_chunks table
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL REFERENCES content(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON document_chunks(file_id);
        """
        
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(create_table_sql)
                    await conn.commit()
                    self.logger.info("Content and document_chunks tables created successfully")
                    return True
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            return False
        
    async def save_content_db(self, file_name: str, object_key: str) -> str:
        """
        Log download metadata to the content table.

        Returns the UUID of the inserted record.
        """
        try:
            # Generate UUID based on file path (deterministic)
            file_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, object_key))

            insert_sql = """
            INSERT INTO content (id, file_name, object_key)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
            """

            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(insert_sql, (file_uuid, file_name, object_key))
                    result = await cursor.fetchone()
                    await conn.commit()

                    if result:
                        record_id = result[0]
                    else:
                        record_id = file_uuid  # UUID already exists
                    
                    self.logger.info(f"Download metadata logged with ID: {record_id}")
                    return record_id
        except Exception as e:
            self.logger.error(f"Error logging download metadata: {e}")
            return None

    async def save_chunk_embeddings(self, chunks, embeddings, file_id):
        """
        Save chunked documents + embeddings to document_chunks table.
        
        :param chunks: Langchain Document chunks
        :param embeddings: List of embedding vectors
        :param file_id: ID from content table
        """
        insert_sql = """
        INSERT INTO document_chunks (id, file_id, chunk_text, metadata)
        VALUES (%s, %s, %s, %s);
        """

        def clean_metadata_for_chroma(metadata_dict):
            """Clean metadata to remove None values and ensure compatible types for ChromaDB"""
            cleaned = {}
            for key, value in metadata_dict.items():
                if value is not None:
                    # Convert to appropriate types that ChromaDB supports
                    if isinstance(value, (str, int, float, bool)):
                        cleaned[key] = value
                    elif isinstance(value, (list, dict)):
                        # Convert complex types to string representation
                        cleaned[key] = str(value)
                    else:
                        # Convert other types to string
                        cleaned[key] = str(value)
                # Skip None values entirely
            return cleaned

        try:
            ids, docs, embeds, chroma_metas = [], [], [], []

            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    for chunk, embed in zip(chunks, embeddings):
                        # Chunk-level unique ID
                        chunk_id = str(uuid.uuid4())

                        # Metadata for Postgres (can include None values)
                        postgres_metadata = {
                            "file_id": file_id,
                            "file_name": chunk.metadata.get("file_name"),
                            "page": chunk.metadata.get("page"),
                            "source": chunk.metadata.get("source")
                        }

                        # Metadata for ChromaDB (cleaned of None values)
                        chroma_metadata = clean_metadata_for_chroma(postgres_metadata)

                        # Insert into Postgres (with all metadata including None values)
                        await cur.execute(
                            insert_sql,
                            (chunk_id, file_id, chunk.page_content, json.dumps(postgres_metadata, default=str))
                        )

                        # Collect for Chroma (with cleaned metadata)
                        ids.append(chunk_id)
                        docs.append(chunk.page_content)
                        embeds.append(embed)
                        chroma_metas.append(chroma_metadata)

                await conn.commit()

            # Save into Chroma with cleaned metadata
            self.logger.info(f"Inserting {len(ids)} chunks into Chroma")
            print("ids:", type(ids), len(ids))
            print("docs:", type(docs), len(docs), type(docs[0]) if docs else None)
            print("embeds:", type(embeds), len(embeds))
            print("metas:", type(chroma_metas), len(chroma_metas))

            collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeds,
                metadatas=chroma_metas,
            )

            self.logger.info(f"File {file_id} stored: {len(chunks)} chunks (Postgres ids + Chroma embeddings persisted)")
            print(f"File {file_id} stored: {len(chunks)} chunks (Postgres ids + Chroma embeddings persisted)")

        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")
            raise

    def retrieve_from_chroma(self, query: str, file_ids: List[str], top_k: int = 5):
        """
        Retrieve top-k chunks from Chroma filtered by file_ids.
        
        Note: This method is synchronous as ChromaDB client is synchronous
        """
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"file_id": {"$in": file_ids}},  # filter by our UUIDs
            )

            # Chroma returns parallel arrays; reshape to dicts
            docs = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for idx in range(len(results["ids"][0])):
                    docs.append({
                        "id": results["ids"][0][idx],
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx]
                    })
            
            self.logger.info(f"Retrieved {len(docs)} chunks from Chroma for query: {query[:50]}...")
            return docs
        except Exception as e:
            self.logger.error(f"Error retrieving from Chroma: {e}")
            return []

    async def get_file_by_id(self, file_id: str):
        """Get file metadata by ID"""
        sql = "SELECT * FROM content WHERE id = %s;"
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (file_id,))
                    row = await cur.fetchone()
                    if row:
                        return {
                            "id": row[0],
                            "file_name": row[1],
                            "object_key": row[2],
                            "downloaded_on": row[3]
                        }
                    return None
        except Exception as e:
            self.logger.error(f"Error getting file by ID: {e}")
            return None

    async def list_all_files(self):
        """List all uploaded files"""
        sql = "SELECT id, file_name, object_key, downloaded_on FROM content ORDER BY downloaded_on DESC;"
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql)
                    rows = await cur.fetchall()
                    return [
                        {
                            "id": row[0],
                            "file_name": row[1],
                            "object_key": row[2],
                            "downloaded_on": row[3].isoformat() if row[3] else None
                        }
                        for row in rows
                    ]
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            return []

    async def get_file_ids_by_names(self, file_names: List[str]):
        """Get file IDs by file names"""
        sql = """
        SELECT file_name, id
        FROM content
        WHERE file_name = ANY(%s);
        """
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (file_names,))
                    rows = await cur.fetchall()
                    mapping = {row[0]: str(row[1]) for row in rows}
                    return mapping
        except Exception as e:
            self.logger.error(f"Error getting file IDs by names: {e}")
            return {}

    async def delete_file(self, file_id: str) -> bool:
        """Delete file and its chunks"""
        try:
            # Delete from Chroma first
            try:
                collection.delete(where={"file_id": file_id})
                self.logger.info(f"Deleted chunks for file {file_id} from Chroma")
            except Exception as chroma_error:
                self.logger.warning(f"Error deleting from Chroma: {chroma_error}")

            # Delete from Postgres (CASCADE will delete chunks)
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM content WHERE id = %s;", (file_id,))
                    await conn.commit()
                    self.logger.info(f"Deleted file {file_id} from database")
                    return True
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            return False

    async def close_pool(self):
        """Close the connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("Connection pool closed")