from typing import List
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import os
from datetime import datetime
from psycopg2 import pool
import psycopg2.extras
from contextlib import contextmanager
from chromadb import Client
from chromadb.config import Settings
import uuid
import psycopg2.extras
import json

# Initialize Chroma client (persistent)
import chromadb

client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("document_chunks")

load_dotenv(".env")

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name=os.getenv("DB_NAME")


DATABASE_URL = f"postgresql://{db_user}:{db_password}@localhost/{db_name}"

class __DatabaseManager:
    def __init__(self):
        """
        Initialize database manager with configuration
        
        :param db_config: Database configuration dictionary
        """
        self.db_config = {
            "host": "localhost",
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": 5432
        }
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)

    def initialize_pool(self, min_conn: int = 1, max_conn: int = 10) -> bool:
        """Initialize the database connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                **self.db_config
            )
            self.logger.info("Database connection pool created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating connection pool: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            if not self.connection_pool:
                raise Exception("Connection pool not initialized")
            
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)

    def create_content_table(self) -> bool:
        """Create the content table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS content (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            object_key TEXT NOT NULL,
            downloaded_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_content_file_name ON content(file_name);
        CREATE INDEX IF NOT EXISTS idx_content_downloaded_on ON content(downloaded_on);
        CREATE INDEX IF NOT EXISTS idx_content_object_key ON content(object_key);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()
                    self.logger.info("Content table created successfully")
                    return True
        except Exception as e:
            self.logger.error(f"Error creating content table: {e}")
            return False
        
    def save_content_db(self, file_name: str, object_key: str) -> str:
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

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(insert_sql, (file_uuid, file_name, object_key))
                    result = cursor.fetchone()
                    conn.commit()

                    if result:
                        record_id = result['id']
                    else:
                        record_id = file_uuid  # UUID already exists
                    self.logger.info(f"Download metadata logged with ID: {record_id}")
                    return record_id
        except Exception as e:
            self.logger.error(f"Error logging download metadata: {e}")
            return None

    def save_chunk_embeddings(self, chunks, embeddings, file_id):
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

            with self.get_connection() as conn:
                with conn.cursor() as cur:
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
                        cur.execute(
                            insert_sql,
                            (chunk_id, file_id, chunk.page_content, json.dumps(postgres_metadata, default=str))
                        )

                        # Collect for Chroma (with cleaned metadata)
                        ids.append(chunk_id)
                        docs.append(chunk.page_content)
                        embeds.append(embed)
                        chroma_metas.append(chroma_metadata)

                conn.commit()

            # Save into Chroma with cleaned metadata
            print("ids:", type(ids), len(ids))
            print("docs:", type(docs), len(docs), type(docs[0]))
            print("embeds:", type(embeds), len(embeds))
            print("metas:", type(chroma_metas), len(chroma_metas))

            collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeds,
                metadatas=chroma_metas,
            )

            logging.info(f"File {file_id} stored: {len(chunks)} chunks (Postgres ids + Chroma embeddings persisted)")
            print(f"File {file_id} stored: {len(chunks)} chunks (Postgres ids + Chroma embeddings persisted)")

        except Exception as e:
            logging.error(f"Error saving chunks: {e}")
            raise

    def retrieve_from_chroma(self,query: str, file_ids: List[str], top_k: int = 5):
        """
        Retrieve top-k chunks from Chroma filtered by file_ids.
        """
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"file_id": {"$in": file_ids}},  # filter by our UUIDs
        )

        # Chroma returns parallel arrays; reshape to dicts
        docs = []
        for idx in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx]
            })
        return docs


