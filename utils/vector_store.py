"""
Vector Store utility for pgvector database operations.
Handles document embedding, storage, and similarity search.
"""

import logging
import json
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import config
import pg8000.native

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    embedding: Optional[List[float]] = None

class VectorStore:
    """PostgreSQL + pgvector store for document embeddings."""
    
    def __init__(self, table_name: str = "documents"):
        """Initialize the vector store."""
        self.table_name = table_name
        self.connection = None
        self.embedding_dim = config.EMBEDDING_DIMENSION
        self._lock = threading.Lock()
        
    def _get_connection(self):
        """Get or create a database connection."""
        if not self.connection:
            # Parse PostgreSQL URL for pg8000
            import urllib.parse
            url = urllib.parse.urlparse(config.POSTGRES_URL)
            
            self.connection = pg8000.native.Connection(
                user=url.username or 'postgres',
                password=url.password or 'password',
                host=url.hostname or 'localhost',
                port=url.port or 5432,
                database=url.path.lstrip('/') or 'postgres'
            )
        return self.connection
        
    async def initialize(self) -> bool:
        """Initialize the database connection and create tables."""
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Enable pgvector extension
                conn.run("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create the documents table
                conn.run(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        document_id TEXT UNIQUE,
                        content TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for vector similarity search
                try:
                    conn.run(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                except Exception as e:
                    # Index creation might fail if there's no data yet
                    logger.warning(f"Could not create vector index (will retry later): {e}")
                
                logger.info(f"Vector store initialized with table: {self.table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents with their embeddings to the vector store."""
        try:
            with self._lock:
                conn = self._get_connection()
                
                for doc in documents:
                    if not doc.embedding:
                        logger.warning(f"Document {doc.id} has no embedding, skipping")
                        continue
                    
                    # Convert embedding to string format for pgvector
                    embedding_str = '[' + ','.join(map(str, doc.embedding)) + ']'
                    
                    # Convert metadata to JSON string
                    metadata_json = json.dumps(doc.metadata) if doc.metadata else '{}'
                    
                    # Insert or update document
                    conn.run(f"""
                        INSERT INTO {self.table_name} (document_id, content, embedding, metadata)
                        VALUES (:doc_id, :content, :embedding, :metadata)
                        ON CONFLICT (document_id) 
                        DO UPDATE SET 
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            created_at = CURRENT_TIMESTAMP
                    """, doc_id=doc.id, content=doc.content, embedding=embedding_str, metadata=metadata_json)
                
                logger.info(f"Added {len(documents)} documents to vector store")
                return True
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def similarity_search(self, query_embedding: List[float], k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine distance."""
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Convert query embedding to string format
                query_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Calculate distance threshold (distance = 1 - similarity)
                distance_threshold = 1.0 - threshold
                
                # Perform similarity search with cosine distance
                # Using HAVING clause due to pg8000 parameter binding issues with vector operations in WHERE
                results = conn.run(f"""
                    SELECT 
                        document_id,
                        content,
                        metadata,
                        1 - (embedding <=> :query_embedding::vector) as similarity
                    FROM {self.table_name}
                    WHERE embedding IS NOT NULL
                    GROUP BY document_id, content, metadata, embedding
                    HAVING embedding <=> :query_embedding::vector <= :distance_threshold
                    ORDER BY embedding <=> :query_embedding::vector
                    LIMIT :k
                """, query_embedding=query_str, distance_threshold=distance_threshold, k=k)
                
                documents = []
                if results:
                    for row in results:
                        doc_id, content, metadata_obj, similarity = row
                        
                        # Handle metadata - pg8000 automatically parses JSONB to dict
                        if isinstance(metadata_obj, dict):
                            metadata = metadata_obj
                        elif isinstance(metadata_obj, str):
                            metadata = json.loads(metadata_obj)
                        else:
                            metadata = {}
                            
                        documents.append({
                            'document_id': doc_id,
                            'content': content,
                            'metadata': metadata,
                            'similarity': float(similarity)
                        })
                
                logger.info(f"Found {len(documents)} similar documents above threshold {threshold}")
                return documents
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        try:
            with self._lock:
                conn = self._get_connection()
                result = conn.run(f"SELECT COUNT(*) FROM {self.table_name}")
                return result[0][0] if result else 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            with self._lock:
                conn = self._get_connection()
                conn.run(f"DELETE FROM {self.table_name} WHERE document_id = :doc_id", doc_id=document_id)
                logger.info(f"Deleted document: {document_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            with self._lock:
                conn = self._get_connection()
                result = conn.run("SELECT 1")
                return result is not None and len(result) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Close the database connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
        except Exception as e:
            logger.error(f"Error closing connection: {e}") 