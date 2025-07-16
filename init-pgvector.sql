-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the healthee_docs table
CREATE TABLE IF NOT EXISTS healthee_docs (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE,
    content TEXT NOT NULL,
    embedding vector(384),  -- Default for all-MiniLM-L6-v2 model
    metadata JSONB,
    url TEXT,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS healthee_docs_embedding_idx 
ON healthee_docs 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); 