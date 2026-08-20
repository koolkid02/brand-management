-- Supabase/Postgres schema for the brand-management memory layer (phase-2).
-- Applied manually (Supabase SQL editor, or `psql "$DATABASE_URL" -f memory/schema.sql`) --
-- this is a 5-table POC schema, not enough surface to justify a migration framework.
--
-- Vector columns use halfvec(3072) to match gemini-embedding-001's default
-- output size (halfvec indexes up to 4000 dims vs. 2000 for plain vector,
-- so this needs zero Gemini-side truncation parameters).

CREATE EXTENSION IF NOT EXISTS vector;

-- Brands: semantic/episodic stay JSONB (1:1 with brand_memory.py's nested
-- dict shape) -- nothing queries INTO them relationally, so normalizing
-- would add join complexity for no present benefit. insights ARE pulled
-- into their own table below because market_memory's aggregation needs a
-- real GROUP BY over them.
CREATE TABLE IF NOT EXISTS brands (
    brand_id        TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    semantic        JSONB NOT NULL,
    episodic        JSONB NOT NULL DEFAULT '{"history": []}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tag vocabulary. embedding is NULLABLE: a tag minted under --mock (or a
-- real embedding-call failure) has no vector yet. Verified against a live
-- pgvector instance: with the HNSW index in play, `ORDER BY embedding <=>
-- query LIMIT k` silently excludes NULL-embedding rows from the result
-- entirely (not merely sorts them last, as a plain seq scan would) -- so
-- they're simply never a vector match until backfilled
-- (`seed_supabase.py --reembed-missing`), but they ARE still findable via
-- the keyword-overlap fallback (reads pattern_statement text directly, not
-- the embedding column).
CREATE TABLE IF NOT EXISTS tag_library (
    tag_id             TEXT PRIMARY KEY,
    pattern_statement  TEXT NOT NULL,
    embedding          halfvec(3072),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tag_library_embedding_hnsw
    ON tag_library USING hnsw (embedding halfvec_cosine_ops);

-- Per-brand tagged insights, normalized out of brands.episodic because
-- market_memory.group_insights_by_tag_category needs a real GROUP BY here.
CREATE TABLE IF NOT EXISTS insights (
    insight_id      TEXT PRIMARY KEY,
    brand_id        TEXT NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    tag             TEXT NOT NULL REFERENCES tag_library(tag_id),
    category        TEXT NOT NULL,
    observation     TEXT NOT NULL,
    date_observed   DATE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS insights_tag_category_idx ON insights (tag, category);
CREATE INDEX IF NOT EXISTS insights_brand_idx ON insights (brand_id);

-- Market (global) memory: promoted, post-abstraction patterns. Materialized
-- by run_aggregation() same as today's market_memory.json -- NOT a live
-- view over insights, preserving PRD §5's "link severed at write time."
CREATE TABLE IF NOT EXISTS market_memory_patterns (
    pattern_id              TEXT PRIMARY KEY,
    tag                     TEXT NOT NULL REFERENCES tag_library(tag_id),
    category                TEXT NOT NULL,
    pattern_statement       TEXT NOT NULL,
    supporting_brand_count  INT NOT NULL,
    embedding               halfvec(3072),
    promoted_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS market_memory_patterns_embedding_hnsw
    ON market_memory_patterns USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS market_memory_patterns_category_idx
    ON market_memory_patterns (category);

-- Trend memory. embedding computed over `label` only, matching
-- trend_memory.retrieve's current exact behavior.
CREATE TABLE IF NOT EXISTS trend_memory_trends (
    trend_id        TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    label           TEXT NOT NULL,
    description     TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    date_observed   DATE NOT NULL,
    source_note     TEXT NOT NULL,
    embedding       halfvec(3072),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trend_memory_trends_embedding_hnsw
    ON trend_memory_trends USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS trend_memory_trends_category_idx
    ON trend_memory_trends (category);
