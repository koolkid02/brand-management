"""Postgres/pgvector primitives for the Supabase memory backend.

Each function here mirrors one local-JSON operation from brand_memory.py /
market_memory.py / trend_memory.py / outcome_memory.py closely enough that
the calling module's `if get_memory_backend() == "supabase":` branch stays a
thin dispatch, not a rewrite (see memory/schema.sql and the phase-2 plan for
the full design rationale). Callers pass an open `conn` when they need to
share a transaction (e.g. outcome_memory's atomic insight+history+tag
write); otherwise each function opens, commits, and closes its own
connection via memory.db.get_connection().

Vector params (`embedding` / `query_vec` arguments) are plain Python
list[float] -- wrapped in pgvector.HalfVector before binding, since
pgvector-python's psycopg adapter only auto-converts that wrapper type (and
numpy arrays) for halfvec columns, not raw lists.
"""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
from pgvector import HalfVector
from psycopg.types.json import Jsonb

from memory import db as memory_db


@contextmanager
def _connection(conn):
    """Yield `conn` unchanged if given (caller owns commit/close); otherwise
    open, commit, and close a fresh one scoped to this call."""
    if conn is not None:
        yield conn
        return
    with memory_db.get_connection() as new_conn:
        yield new_conn
        new_conn.commit()


def _vec(embedding: list[float] | None) -> HalfVector | None:
    return None if embedding is None else HalfVector(embedding)


# ---------------------------------------------------------------------------
# brands
# ---------------------------------------------------------------------------

def list_brands(conn=None) -> list[str]:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT brand_id FROM brands ORDER BY brand_id")
        return [row["brand_id"] for row in cur.fetchall()]


def load_brand(brand_id: str, conn=None) -> dict:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT brand_id, schema_version, semantic, episodic FROM brands WHERE brand_id = %s",
            (brand_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No brand found with brand_id={brand_id!r}")
        cur.execute(
            """SELECT insight_id, tag, category, observation, date_observed
               FROM insights WHERE brand_id = %s ORDER BY insight_id""",
            (brand_id,),
        )
        insights = [
            {
                "insight_id": r["insight_id"],
                "tag": r["tag"],
                "category": r["category"],
                "observation": r["observation"],
                "date_observed": r["date_observed"].isoformat(),
            }
            for r in cur.fetchall()
        ]
    return {
        "brand_id": row["brand_id"],
        "schema_version": row["schema_version"],
        "semantic": row["semantic"],
        "episodic": row["episodic"],
        "insights": insights,
    }


def load_all_brands(conn=None) -> dict[str, dict]:
    with _connection(conn) as c:
        return {brand_id: load_brand(brand_id, conn=c) for brand_id in list_brands(conn=c)}


def append_insight(brand_id: str, insight: dict, conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO insights (insight_id, brand_id, tag, category, observation, date_observed)
               VALUES (%(insight_id)s, %(brand_id)s, %(tag)s, %(category)s, %(observation)s, %(date_observed)s)""",
            {**insight, "brand_id": brand_id},
        )


def append_history_entry(brand_id: str, history_entry: dict, conn=None) -> None:
    """Read-modify-write in Python then one UPDATE, matching the existing
    local-JSON "mutate the list in Python, write the whole structure back"
    style rather than introducing JSONB path-operator SQL as a new idiom.
    `FOR UPDATE` locks the row for the rest of this transaction so two
    concurrent appends can't race and drop one of them."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT episodic FROM brands WHERE brand_id = %s FOR UPDATE", (brand_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No brand found with brand_id={brand_id!r}")
        episodic = row["episodic"]
        episodic.setdefault("history", []).append(history_entry)
        cur.execute(
            "UPDATE brands SET episodic = %s, updated_at = now() WHERE brand_id = %s",
            (Jsonb(episodic), brand_id),
        )


def append_insight_and_history(brand_id: str, insight: dict, history_entry: dict, conn=None) -> None:
    with _connection(conn) as c:
        append_insight(brand_id, insight, conn=c)
        append_history_entry(brand_id, history_entry, conn=c)


def upsert_brand(brand: dict, conn=None) -> None:
    """Idempotent brand upsert for seed_supabase.py. `insights` live in
    their own table (upsert_insight) -- this only touches
    brands.semantic/episodic."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO brands (brand_id, schema_version, semantic, episodic)
               VALUES (%(brand_id)s, %(schema_version)s, %(semantic)s, %(episodic)s)
               ON CONFLICT (brand_id) DO UPDATE SET
                   schema_version = EXCLUDED.schema_version,
                   semantic = EXCLUDED.semantic,
                   episodic = EXCLUDED.episodic,
                   updated_at = now()""",
            {
                "brand_id": brand["brand_id"],
                "schema_version": brand["schema_version"],
                "semantic": Jsonb(brand["semantic"]),
                "episodic": Jsonb(brand["episodic"]),
            },
        )


def upsert_insight(insight: dict, brand_id: str, conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO insights (insight_id, brand_id, tag, category, observation, date_observed)
               VALUES (%(insight_id)s, %(brand_id)s, %(tag)s, %(category)s, %(observation)s, %(date_observed)s)
               ON CONFLICT (insight_id) DO UPDATE SET
                   brand_id = EXCLUDED.brand_id, tag = EXCLUDED.tag, category = EXCLUDED.category,
                   observation = EXCLUDED.observation, date_observed = EXCLUDED.date_observed""",
            {**insight, "brand_id": brand_id},
        )


# ---------------------------------------------------------------------------
# tag_library
# ---------------------------------------------------------------------------

def load_tag_library(conn=None) -> dict[str, str]:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT tag_id, pattern_statement FROM tag_library ORDER BY tag_id")
        return {row["tag_id"]: row["pattern_statement"] for row in cur.fetchall()}


def register_new_tag(tag_id: str, pattern_statement: str, embedding: list[float] | None = None, conn=None) -> None:
    """Relies on tag_library's PK constraint rather than a check-then-insert
    -- callers (market_memory.resolve_or_create_tag) already disambiguate
    ids against an in-memory tag_library snapshot before calling this, so a
    UniqueViolation here only fires on a genuine concurrent-write race;
    re-raised as the same ValueError shape the local-JSON backend raises."""
    try:
        with _connection(conn) as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO tag_library (tag_id, pattern_statement, embedding) VALUES (%s, %s, %s)",
                (tag_id, pattern_statement, _vec(embedding)),
            )
    except psycopg.errors.UniqueViolation:
        raise ValueError(f"Tag {tag_id!r} is already registered") from None


def match_tag_by_embedding(candidate_vec: list[float], threshold: float, conn=None) -> tuple[str, float] | None:
    """Best cosine match for candidate_vec among tags that HAVE an embedding,
    or None if the best match scores below `threshold`. Tags with a NULL
    embedding (minted under --mock, or a past embedding failure) are
    excluded from the ranked result entirely under the HNSW index scan --
    verified empirically against a live pgvector instance -- so they can
    never falsely win a vector match; they remain reachable only via the
    keyword-overlap fallback in market_memory.resolve_or_create_tag."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """SELECT tag_id, 1 - (embedding <=> %(vec)s) AS similarity
               FROM tag_library
               WHERE embedding IS NOT NULL
               ORDER BY embedding <=> %(vec)s
               LIMIT 1""",
            {"vec": _vec(candidate_vec)},
        )
        row = cur.fetchone()
    if row is None or row["similarity"] < threshold:
        return None
    return row["tag_id"], row["similarity"]


def upsert_tag(tag_id: str, pattern_statement: str, embedding: list[float] | None = None, conn=None) -> None:
    """Idempotent tag upsert for seed_supabase.py. embedding=None preserves
    whatever's already stored (COALESCE) rather than clearing it -- lets a
    `--mock` structural reload run safely after a real `--reembed-missing`
    pass without discarding the embeddings it computed."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO tag_library (tag_id, pattern_statement, embedding)
               VALUES (%(tag_id)s, %(pattern_statement)s, %(embedding)s)
               ON CONFLICT (tag_id) DO UPDATE SET
                   pattern_statement = EXCLUDED.pattern_statement,
                   embedding = COALESCE(EXCLUDED.embedding, tag_library.embedding)""",
            {"tag_id": tag_id, "pattern_statement": pattern_statement, "embedding": _vec(embedding)},
        )


def tags_missing_embedding(conn=None) -> list[dict]:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT tag_id, pattern_statement FROM tag_library WHERE embedding IS NULL ORDER BY tag_id")
        return [dict(row) for row in cur.fetchall()]


def set_tag_embedding(tag_id: str, embedding: list[float], conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("UPDATE tag_library SET embedding = %s WHERE tag_id = %s", (_vec(embedding), tag_id))


# ---------------------------------------------------------------------------
# market_memory_patterns
# ---------------------------------------------------------------------------

def compute_promoted_patterns(min_brands: int = 2, conn=None) -> list[dict]:
    """SQL-native sibling of market_memory.promote_patterns: the literal
    translation of the 2-brand-recurrence rule into GROUP BY ... HAVING."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """SELECT i.tag AS tag, i.category AS category, tl.pattern_statement AS pattern_statement,
                      COUNT(DISTINCT i.brand_id) AS supporting_brand_count
               FROM insights i
               JOIN tag_library tl ON tl.tag_id = i.tag
               GROUP BY i.tag, i.category, tl.pattern_statement
               HAVING COUNT(DISTINCT i.brand_id) >= %(min_brands)s""",
            {"min_brands": min_brands},
        )
        rows = cur.fetchall()
    patterns = [
        {
            "pattern_id": f"{row['category']}__{row['tag']}",
            "tag": row["tag"],
            "category": row["category"],
            "pattern_statement": row["pattern_statement"],
            "supporting_brand_count": row["supporting_brand_count"],
        }
        for row in rows
    ]
    return sorted(patterns, key=lambda p: p["pattern_id"])


def market_pattern_ids_with_embedding(pattern_ids: list[str], conn=None) -> set[str]:
    if not pattern_ids:
        return set()
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT pattern_id FROM market_memory_patterns WHERE pattern_id = ANY(%(ids)s) AND embedding IS NOT NULL",
            {"ids": pattern_ids},
        )
        return {row["pattern_id"] for row in cur.fetchall()}


def market_patterns_missing_embedding(conn=None) -> list[dict]:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT pattern_id, pattern_statement FROM market_memory_patterns WHERE embedding IS NULL ORDER BY pattern_id")
        return [dict(row) for row in cur.fetchall()]


def set_market_pattern_embedding(pattern_id: str, embedding: list[float], conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("UPDATE market_memory_patterns SET embedding = %s WHERE pattern_id = %s", (_vec(embedding), pattern_id))


def upsert_market_pattern(pattern: dict, embedding: list[float] | None = None, conn=None) -> None:
    """embedding=None preserves whatever's already stored (COALESCE) --
    the mechanism behind "embeddings computed once, persisted, reused":
    run_aggregation only passes a real embedding for patterns that don't
    already have one (see market_pattern_ids_with_embedding)."""
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO market_memory_patterns
                   (pattern_id, tag, category, pattern_statement, supporting_brand_count, embedding)
               VALUES (%(pattern_id)s, %(tag)s, %(category)s, %(pattern_statement)s, %(supporting_brand_count)s, %(embedding)s)
               ON CONFLICT (pattern_id) DO UPDATE SET
                   tag = EXCLUDED.tag,
                   category = EXCLUDED.category,
                   pattern_statement = EXCLUDED.pattern_statement,
                   supporting_brand_count = EXCLUDED.supporting_brand_count,
                   embedding = COALESCE(EXCLUDED.embedding, market_memory_patterns.embedding)""",
            {**pattern, "embedding": _vec(embedding)},
        )


def load_market_patterns(category: str | None = None, conn=None) -> list[dict]:
    """Raw pattern rows, no vector math -- used by market_memory.retrieve's
    mock-mode branch (keyword_overlap over the full matching-category set,
    same as the local-JSON backend)."""
    sql = "SELECT pattern_id, tag, category, pattern_statement, supporting_brand_count FROM market_memory_patterns"
    params: dict = {}
    if category is not None:
        sql += " WHERE category = %(category)s"
        params["category"] = category
    sql += " ORDER BY pattern_id"
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def retrieve_market_patterns(query_vec: list[float], category: str | None = None, top_k: int = 3, conn=None) -> list[dict]:
    """SQL-side category filter + vector rank + LIMIT in one statement --
    safe to truncate here because similarity alone determines market_memory's
    ranking (unlike trend_memory, there's no post-hoc decay factor)."""
    where = ["embedding IS NOT NULL"]
    params: dict = {"vec": _vec(query_vec), "top_k": top_k}
    if category is not None:
        where.append("category = %(category)s")
        params["category"] = category
    sql = f"""
        SELECT pattern_id, tag, category, pattern_statement, supporting_brand_count,
               1 - (embedding <=> %(vec)s) AS similarity
        FROM market_memory_patterns
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %(vec)s
        LIMIT %(top_k)s
    """
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# trend_memory_trends
# ---------------------------------------------------------------------------

def load_trends(category: str | None = None, conn=None) -> list[dict]:
    """Raw trend rows, no vector math -- used by trend_memory.retrieve's
    mock-mode branch."""
    sql = "SELECT trend_id, category, label, description, signal_type, date_observed, source_note FROM trend_memory_trends"
    params: dict = {}
    if category is not None:
        sql += " WHERE category = %(category)s"
        params["category"] = category
    sql += " ORDER BY trend_id"
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["date_observed"] = row["date_observed"].isoformat()
    return rows


def retrieve_trend_candidates(query_vec: list[float], category: str | None = None, conn=None) -> list[dict]:
    """SQL-side category filter + vector similarity, deliberately NO LIMIT:
    trend_memory's ranking multiplies similarity by a freshness decay that's
    a function of "today" and can't be indexed or precomputed, so final
    top_k truncation must happen in Python after combining both factors
    (matching trend_memory.retrieve's existing design) -- truncating in SQL
    here could wrongly discard a less-similar-but-fresher trend."""
    where = ["embedding IS NOT NULL"]
    params: dict = {"vec": _vec(query_vec)}
    if category is not None:
        where.append("category = %(category)s")
        params["category"] = category
    sql = f"""
        SELECT trend_id, category, label, description, signal_type, date_observed, source_note,
               1 - (embedding <=> %(vec)s) AS similarity
        FROM trend_memory_trends
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %(vec)s
    """
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["date_observed"] = row["date_observed"].isoformat()
    return rows


def upsert_trend(trend: dict, embedding: list[float] | None = None, conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO trend_memory_trends
                   (trend_id, category, label, description, signal_type, date_observed, source_note, embedding)
               VALUES (%(trend_id)s, %(category)s, %(label)s, %(description)s, %(signal_type)s,
                       %(date_observed)s, %(source_note)s, %(embedding)s)
               ON CONFLICT (trend_id) DO UPDATE SET
                   category = EXCLUDED.category, label = EXCLUDED.label, description = EXCLUDED.description,
                   signal_type = EXCLUDED.signal_type, date_observed = EXCLUDED.date_observed,
                   source_note = EXCLUDED.source_note,
                   embedding = COALESCE(EXCLUDED.embedding, trend_memory_trends.embedding)""",
            {**trend, "embedding": _vec(embedding)},
        )


def trends_missing_embedding(conn=None) -> list[dict]:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("SELECT trend_id, label FROM trend_memory_trends WHERE embedding IS NULL ORDER BY trend_id")
        return [dict(row) for row in cur.fetchall()]


def set_trend_embedding(trend_id: str, embedding: list[float], conn=None) -> None:
    with _connection(conn) as c, c.cursor() as cur:
        cur.execute("UPDATE trend_memory_trends SET embedding = %s WHERE trend_id = %s", (_vec(embedding), trend_id))
