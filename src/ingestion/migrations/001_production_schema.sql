-- ===========================================================================
-- ShasanAI Production Database Schema Migration (001_production_schema.sql)
-- Designed for 40,000+ Government Order PDFs RAG Scaling
-- PostgreSQL 16 + pgvector with Full-Text Search (tsvector) and Composite B-Trees
-- ===========================================================================

-- 1. Ensure pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Parent Documents Master Table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    document_id TEXT UNIQUE NOT NULL,
    go_number TEXT NOT NULL,
    issuing_department TEXT NOT NULL,
    issuing_authority TEXT DEFAULT 'उत्तराखण्ड शासन',
    date DATE NOT NULL,
    year INT NOT NULL,
    total_pages INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'CURRENT_ACTIVE', -- CURRENT_ACTIVE, SUPERSEDED, AMENDED, REPEALED
    subject TEXT,
    file_path TEXT NOT NULL,
    ocr_quality_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_docs_go ON documents(go_number);
CREATE INDEX IF NOT EXISTS idx_docs_dept_year ON documents(issuing_department, year);
CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_docs_filepath ON documents(file_path);

-- 3. Supersession Relational Graph Table
CREATE TABLE IF NOT EXISTS supersession_graph (
    id SERIAL PRIMARY KEY,
    go_number TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'CURRENT_ACTIVE', -- CURRENT_ACTIVE, SUPERSEDED, AMENDED
    superseded_by TEXT,
    amends TEXT,
    effective_date DATE,
    gazette_notification_ref TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_supersession_go ON supersession_graph(go_number);
CREATE INDEX IF NOT EXISTS idx_supersession_status ON supersession_graph(status);

-- 4. Document Chunks Schema Upgrades (with generated tsvector column for FTS)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    file_path TEXT DEFAULT NULL,
    go_number TEXT NOT NULL,
    issuing_department TEXT NOT NULL,
    date TEXT NOT NULL,
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,
    exact_text_excerpt TEXT NOT NULL,
    bounding_box_coordinates JSONB DEFAULT NULL,
    table_json JSONB DEFAULT NULL,
    math_verification_status JSONB DEFAULT NULL,
    font_encoding_type TEXT DEFAULT NULL,
    embedding vector(3072),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Idempotent column additions
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS year INT,
    ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'CURRENT_ACTIVE',
    ADD COLUMN IF NOT EXISTS bounding_box_coordinates JSONB DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS file_path TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS table_json JSONB DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS math_verification_status JSONB DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS font_encoding_type TEXT DEFAULT NULL;

-- Idempotent vector dimension alteration (upgrade to 3072 for Google GenAI embeddings)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_chunks' AND column_name = 'embedding'
    ) THEN
        BEGIN
            DROP INDEX IF EXISTS document_chunks_embedding_idx;
            ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(3072);
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END IF;
END $$;

-- Add generated tsvector column if not exists (using 'simple' configuration to preserve multilingual Devanagari tokens)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_chunks' AND column_name = 'tsv_content'
    ) THEN
        ALTER TABLE document_chunks 
        ADD COLUMN tsv_content tsvector 
        GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(exact_text_excerpt, ''))) STORED;
    END IF;
END $$;

-- 5. Indexes for 4-Layer RAG Retrieval Funnel
DO $$
BEGIN
    -- pgvector HNSW index supports maximum 2,000 dimensions
    IF EXISTS (
        SELECT 1 FROM pg_attribute 
        WHERE attrelid = 'document_chunks'::regclass 
          AND attname = 'embedding' 
          AND atttypmod <= 2000
    ) THEN
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON document_chunks USING gin(tsv_content);

CREATE INDEX IF NOT EXISTS idx_chunks_dept_year
    ON document_chunks (issuing_department, year);

CREATE INDEX IF NOT EXISTS idx_chunks_status
    ON document_chunks (status);

CREATE INDEX IF NOT EXISTS idx_chunks_go_num
    ON document_chunks (go_number);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_page
    ON document_chunks (document_id, page_number, chunk_index);

CREATE INDEX IF NOT EXISTS document_chunks_filepath_idx
    ON document_chunks (file_path);
