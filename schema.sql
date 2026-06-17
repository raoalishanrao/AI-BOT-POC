-- Run this in Supabase SQL Editor before ingest.py

DROP TABLE IF EXISTS university_knowledge CASCADE;
DROP FUNCTION IF EXISTS update_fts_vector() CASCADE;

CREATE TABLE university_knowledge (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    meta_data JSONB,
    fts_tokens TSVECTOR
);

CREATE OR REPLACE FUNCTION update_fts_vector()
RETURNS trigger AS $$
BEGIN
    new.fts_tokens := to_tsvector(
        'english',
        COALESCE(new.content, '') || ' ' || COALESCE(new.meta_data->>'keywords', '')
    );
    RETURN new;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_knowledge_fts
BEFORE INSERT OR UPDATE ON university_knowledge
FOR EACH ROW EXECUTE FUNCTION update_fts_vector();

CREATE INDEX knowledge_fts_idx ON university_knowledge USING GIN(fts_tokens);

CREATE TABLE IF NOT EXISTS chat_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL DEFAULT 'Prospective Student',
    phone_number VARCHAR(50) UNIQUE,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- If chat_users already exists with NOT NULL phone_number, run in Supabase SQL Editor:
-- ALTER TABLE chat_users ALTER COLUMN phone_number DROP NOT NULL;
-- CREATE UNIQUE INDEX IF NOT EXISTS chat_users_email_unique ON chat_users(email) WHERE email IS NOT NULL;

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES chat_users(id) ON DELETE SET NULL,
    device_info JSONB,
    profile JSONB DEFAULT '{}'::jsonb,
    stage VARCHAR(50) DEFAULT 'introduction',
    lead_status VARCHAR(50) DEFAULT 'new',
    recommended_programs JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Run if chat_sessions already exists without counselor columns:
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS profile JSONB DEFAULT '{}'::jsonb;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS stage VARCHAR(50) DEFAULT 'introduction';
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS lead_status VARCHAR(50) DEFAULT 'new';
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS recommended_programs JSONB DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS counselor_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(session_id) ON DELETE SET NULL,
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    interested_program VARCHAR(255),
    preferred_contact_time VARCHAR(100),
    lead_score INT DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE OR REPLACE FUNCTION match_university_knowledge_hybrid (
    query_embedding VECTOR(768),
    query_text TEXT,
    match_count INT
)
RETURNS TABLE (
    id INT,
    content TEXT,
    meta_data JSONB,
    vector_similarity FLOAT,
    fts_rank REAL,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH scored_records AS (
        SELECT
            uk.id,
            uk.content,
            uk.meta_data,
            (1 - (uk.embedding <=> query_embedding)) AS v_sim,
            ts_rank_cd(uk.fts_tokens, websearch_to_tsquery('english', query_text)) AS f_rank
        FROM university_knowledge uk
    )
    SELECT
        sr.id,
        sr.content,
        sr.meta_data,
        sr.v_sim AS vector_similarity,
        sr.f_rank AS fts_rank,
        ((0.7 * sr.v_sim) + (0.3 * COALESCE(sr.f_rank, 0)))::FLOAT AS combined_score
    FROM scored_records sr
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;
