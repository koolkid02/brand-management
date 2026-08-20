"""Connection ownership for the Supabase/Postgres memory backend.

psycopg v3 (not psycopg2/SQLAlchemy/asyncpg): this codebase is synchronous
and plain-dict-in/dict-out everywhere with zero ORM precedent, which
`dict_row` matches natively. It also composes for free with a future
`langgraph-checkpoint-postgres` PostgresSaver swap in agent_graph.py, which
itself depends on psycopg v3 and the same DATABASE_URL.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config import get_database_url


def get_connection() -> psycopg.Connection:
    """Caller owns the connection lifecycle (use as a context manager) and
    the transaction boundary (explicit commit/rollback -- autocommit stays
    off by default). `prepare_threshold=None` disables server-side prepared
    statements, required for compatibility with Supabase's Supavisor pooler
    in transaction mode."""
    conn = psycopg.connect(get_database_url(), row_factory=dict_row, prepare_threshold=None)
    register_vector(conn)  # halfvec/vector param + result adapters (pgvector.HalfVector <-> Postgres)
    return conn
