-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core documents table for RAG
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,

    -- Content
    content TEXT NOT NULL,
    embedding vector(768),              -- nomic-embed-text produces 768-dim vectors

    -- Metadata for filtering
    ticker VARCHAR(10),                 -- NULL for macro/general docs
    doc_type VARCHAR(20) NOT NULL,      -- 'news', 'filing_10k', 'filing_10q', 'filing_8k'
    source VARCHAR(50) NOT NULL,        -- 'alpaca', 'sec_edgar', 'manual'
    source_url TEXT,                    -- Original URL for citation/dedup

    -- Timestamps
    published_at TIMESTAMP NOT NULL,    -- When the source was published
    created_at TIMESTAMP DEFAULT NOW(), -- When we ingested it

    -- Chunking metadata
    parent_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER DEFAULT 0
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_documents_embedding ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filtering indexes
CREATE INDEX idx_documents_ticker ON documents(ticker);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_published_at ON documents(published_at DESC);
CREATE INDEX idx_documents_ticker_published ON documents(ticker, published_at DESC);
CREATE INDEX idx_documents_source_url ON documents(source_url);
