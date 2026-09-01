"""PostgreSQL 16 pgvector store and enterprise hybrid search engine for Government Orders.

Integrates local bge-m3 embeddings (1024-dim), PostgreSQL Full-Text Search (tsvector),
SQL-level Reciprocal Rank Fusion (RRF, k=60), and pooled async/sync connection management.
"""

import asyncio
import decimal
import json
import logging
import math
import os
import re
from typing import Any, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from dotenv import load_dotenv

from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
from src.ingestion.chunking import DocumentChunk
from src.ingestion.schema_migration import apply_migrations
from src.state.checkpointing import ensure_windows_event_loop
from src.tools.schemas.search_go_corpus import PassageMatch
from src.utils.dept_mapper import get_dept_keywords
from src.utils.model_runtime import get_embeddings_model

logger = logging.getLogger("shasanai.vector_store")
load_dotenv()
ensure_windows_event_loop()

DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shasanai")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))


def _ensure_embedding_dimension(conn: Any, target_dim: int = 3072) -> None:
    """Checks and updates the embedding column vector dimension if mismatched."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT atttypmod FROM pg_attribute 
                WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';
                """
            )
            row = cur.fetchone()
            if row and row.get("atttypmod") is not None and row["atttypmod"] != target_dim:
                logger.info("Migrating document_chunks.embedding dimension from %s to %s", row["atttypmod"], target_dim)
                cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_idx;")
                cur.execute(f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({target_dim});")
                if target_dim <= 2000:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                        ON document_chunks USING hnsw (embedding vector_cosine_ops);
                        """
                    )
    except Exception as exc:
        logger.warning("Could not auto-migrate vector dimension (%s): %s", type(exc).__name__, exc)


def sanitize_relevance_score(raw_score: Any) -> float:
    """Sanitizes relevance score, replacing NaN/None with 0.0 and clamping to [0.0, 1.0]."""
    if raw_score is None:
        return 0.0
    try:
        val = float(raw_score)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.0


def decimal_default(obj: Any) -> Any:
    """Serializes Decimal objects to float, and pydantic/custom models to dict for JSONB insertion."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def extract_year_from_date(date_str: str | None) -> int:
    """Extracts 4-digit year as integer from date string (e.g. '2018-03-12' -> 2018)."""
    if not date_str:
        return 2018
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(date_str))
    if match:
        return int(match.group(1))
    return 2018


class VectorStore:
    """Enterprise pgvector store managing connection pools, migrations, and SQL-level RRF hybrid search."""

    _async_pool: Optional[AsyncConnectionPool] = None
    _sync_pool: Optional[ConnectionPool] = None
    _pool_lock = asyncio.Lock()

    def __init__(self, db_url: str = DEFAULT_DATABASE_URL) -> None:
        self.db_url = db_url
        self._embeddings = get_embeddings_model()
        self._init_sync_pool()

    def _init_sync_pool(self) -> None:
        """Initializes class-level sync ConnectionPool singleton."""
        if VectorStore._sync_pool is None:
            ensure_windows_event_loop()
            try:
                VectorStore._sync_pool = ConnectionPool(
                    conninfo=self.db_url,
                    min_size=2,
                    max_size=10,
                    timeout=30.0,
                    open=True,
                    kwargs={"autocommit": True, "row_factory": dict_row},
                )
            except Exception as exc:
                logger.warning("Could not initialize sync connection pool: %s", exc)

    @property
    def sync_pool(self) -> ConnectionPool:
        """Returns the active sync ConnectionPool."""
        if VectorStore._sync_pool is None:
            self._init_sync_pool()
        if VectorStore._sync_pool is None:
            raise RuntimeError("Sync database connection pool is uninitialized.")
        return VectorStore._sync_pool

    async def get_async_pool(self) -> AsyncConnectionPool:
        """Returns the active AsyncConnectionPool singleton with thread-safe async initialization."""
        ensure_windows_event_loop()
        if VectorStore._async_pool is None or VectorStore._async_pool.closed:
            async with VectorStore._pool_lock:
                if VectorStore._async_pool is None or VectorStore._async_pool.closed:
                    pool = AsyncConnectionPool(
                        conninfo=self.db_url,
                        min_size=4,
                        max_size=30,
                        timeout=30.0,
                        open=False,
                        kwargs={"autocommit": True, "row_factory": dict_row},
                    )
                    await pool.open()
                    VectorStore._async_pool = pool
        return VectorStore._async_pool

    @classmethod
    async def close_pools(cls) -> None:
        """Gracefully closes active sync and async connection pools."""
        if cls._async_pool and not cls._async_pool.closed:
            try:
                await cls._async_pool.close()
            except Exception:
                pass
            cls._async_pool = None
        if cls._sync_pool and not cls._sync_pool.closed:
            try:
                cls._sync_pool.close()
            except Exception:
                pass
            cls._sync_pool = None

    @property
    def pool(self) -> AsyncConnectionPool:
        """Convenience property for AsyncConnectionPool."""
        if VectorStore._async_pool is None:
            ensure_windows_event_loop()
            pool = AsyncConnectionPool(
                conninfo=self.db_url,
                min_size=4,
                max_size=30,
                timeout=30.0,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            # Open if event loop is available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(pool.open())
                else:
                    loop.run_until_complete(pool.open())
            except Exception:
                pass
            VectorStore._async_pool = pool
        return VectorStore._async_pool

    def get_connection(self):
        """Returns a connection context manager from the sync pool."""
        return self.sync_pool.connection()

    async def get_async_connection(self):
        """Returns an async connection context manager from the async pool."""
        p = await self.get_async_pool()
        return p.connection()

    def initialize_schema(self) -> None:
        """Applies idempotent production migrations for documents, supersession, and tsvector columns."""
        try:
            apply_migrations(self.db_url)
            with self.get_connection() as conn:
                _ensure_embedding_dimension(conn, target_dim=EMBEDDING_DIMENSION)
            logger.info("Database schema migrations verified.")
        except Exception as exc:
            logger.error("Schema migration failed: %s", exc)
            raise

    def clear_all_chunks(self) -> int:
        """Truncates document_chunks and documents tables to wipe legacy data."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE document_chunks, documents, supersession_graph CASCADE;")
                    return cur.rowcount
        except Exception:
            return 0

    async def a_clear_all_chunks(self) -> int:
        """Asynchronously truncates document_chunks table using worker thread."""
        return await asyncio.to_thread(self.clear_all_chunks)

    def count_chunks(self) -> int:
        """Synchronously returns total number of document chunks in the database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) AS total FROM document_chunks;")
                    row = cur.fetchone()
                    return row["total"] if row else 0
        except Exception:
            return 0

    async def a_count_chunks(self) -> int:
        """Asynchronously returns total number of document chunks in the database."""
        return await asyncio.to_thread(self.count_chunks)

    def print_sample_chunks(self, n: int = 3) -> None:
        """Prints n sample chunks from the database for post-ingestion verification."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            chunk_id,
                            go_number,
                            year,
                            status,
                            page_number,
                            file_path,
                            (bounding_box_coordinates IS NOT NULL) AS has_bbox,
                            LEFT(exact_text_excerpt, 120) AS preview
                        FROM document_chunks
                        ORDER BY id ASC
                        LIMIT %s;
                        """,
                        (n,),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            print(f"  [sample query failed: {exc}]")
            return

        if not rows:
            print("  (no chunks found in database)")
            return

        for i, row in enumerate(rows, start=1):
            bbox_tag = "✓ bbox" if row["has_bbox"] else "✗ no-bbox"
            print(
                f"  Sample {i}: [{row['chunk_id']}] "
                f"GO={row['go_number']} | Year={row.get('year')} | Status={row.get('status')} | p{row['page_number']} | {bbox_tag}"
            )
            print(f"    Text: {row['preview']!r}")

    def insert_document_record(
        self,
        document_id: str,
        go_number: str,
        issuing_department: str,
        issuing_authority: str = "उत्तराखण्ड शासन",
        date: str = "2018-01-01",
        total_pages: int = 1,
        status: str = "CURRENT_ACTIVE",
        subject: Optional[str] = None,
        file_path: str = "",
        ocr_quality_score: float = 1.0,
    ) -> None:
        """Inserts or updates parent document record in the documents master table."""
        year = extract_year_from_date(date)
        clean_date = date if re.match(r"^\d{4}-\d{2}-\d{2}$", date) else f"{year}-01-01"

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (
                        document_id, go_number, issuing_department, issuing_authority,
                        date, year, total_pages, status, subject, file_path, ocr_quality_score
                    ) VALUES (%s, %s, %s, %s, %s::date, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        go_number = EXCLUDED.go_number,
                        issuing_department = EXCLUDED.issuing_department,
                        date = EXCLUDED.date,
                        year = EXCLUDED.year,
                        total_pages = EXCLUDED.total_pages,
                        status = EXCLUDED.status,
                        subject = EXCLUDED.subject,
                        file_path = EXCLUDED.file_path,
                        ocr_quality_score = EXCLUDED.ocr_quality_score;
                    """,
                    (
                        document_id,
                        go_number,
                        issuing_department,
                        issuing_authority,
                        clean_date,
                        year,
                        total_pages,
                        status,
                        subject,
                        file_path,
                        ocr_quality_score,
                    ),
                )

    def insert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 4) -> int:
        """Generates embeddings and inserts chunks into PostgreSQL in micro-batches with year, status, and FTS support."""
        import time

        if not chunks:
            return 0

        self.initialize_schema()
        inserted_count = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.exact_text_excerpt for c in batch]

            embeddings: list[list[float]] = []
            for t in texts:
                clean_t = DevanagariNormalizer.normalize_text(t.strip())
                if not clean_t:
                    embeddings.append([0.0] * EMBEDDING_DIMENSION)
                else:
                    try:
                        emb_vec = self._embeddings.embed_query(clean_t[:1500])
                        embeddings.append(emb_vec)
                    except Exception:
                        embeddings.append([0.0] * EMBEDDING_DIMENSION)

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk, emb in zip(batch, embeddings):
                        emb_str = f"[{','.join(str(x) for x in emb)}]"
                        norm_excerpt = DevanagariNormalizer.normalize_text(chunk.exact_text_excerpt)
                        norm_go = DevanagariNormalizer.normalize_text(chunk.go_number)
                        chunk_year = extract_year_from_date(chunk.date)

                        bbox_json: Optional[str] = None
                        if chunk.bounding_box_coordinates is not None:
                            bbox_json = json.dumps(chunk.bounding_box_coordinates, default=decimal_default)

                        tbl_json: Optional[str] = None
                        if getattr(chunk, "table_json", None) is not None:
                            tbl_json = json.dumps(chunk.table_json, default=decimal_default)

                        math_json: Optional[str] = None
                        if getattr(chunk, "math_verification_status", None) is not None:
                            math_json = json.dumps(chunk.math_verification_status, default=decimal_default)

                        font_enc: Optional[str] = getattr(chunk, "font_encoding_type", None)

                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                chunk_id, document_id, file_path, go_number, issuing_department,
                                date, year, status, page_number, chunk_index, exact_text_excerpt,
                                bounding_box_coordinates, table_json, math_verification_status, font_encoding_type, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'CURRENT_ACTIVE', %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::vector)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                file_path = EXCLUDED.file_path,
                                page_number = EXCLUDED.page_number,
                                chunk_index = EXCLUDED.chunk_index,
                                exact_text_excerpt = EXCLUDED.exact_text_excerpt,
                                bounding_box_coordinates = EXCLUDED.bounding_box_coordinates,
                                table_json = EXCLUDED.table_json,
                                math_verification_status = EXCLUDED.math_verification_status,
                                font_encoding_type = EXCLUDED.font_encoding_type,
                                embedding = EXCLUDED.embedding,
                                go_number = EXCLUDED.go_number,
                                issuing_department = EXCLUDED.issuing_department,
                                year = EXCLUDED.year,
                                date = EXCLUDED.date;
                            """,
                            (
                                chunk.chunk_id,
                                chunk.document_id,
                                chunk.file_path,
                                norm_go,
                                chunk.issuing_department,
                                chunk.date,
                                chunk_year,
                                chunk.page_number,
                                chunk.chunk_index,
                                norm_excerpt,
                                bbox_json,
                                tbl_json,
                                math_json,
                                font_enc,
                                emb_str,
                            ),
                        )
                        inserted_count += 1

            time.sleep(0.05)

        return inserted_count

    async def a_generate_embedding(self, text: str) -> list[float]:
        """Generates embedding vector asynchronously."""
        clean_text = DevanagariNormalizer.normalize_text(text.strip())
        if not clean_text:
            return [0.0] * EMBEDDING_DIMENSION
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._embeddings.embed_query, clean_text[:1500]),
                timeout=3.0,
            )
        except Exception as exc:
            logger.warning("Embedding generation failed (%s): %s", type(exc).__name__, exc)
            return [0.0] * EMBEDDING_DIMENSION

    async def a_hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[list[float]] = None,
        department_filter: Optional[str] = None,
        year_range_filter: Optional[tuple[int, int] | list[int]] = None,
        go_number_filter: Optional[str] = None,
        status_filter: Optional[str] = "CURRENT_ACTIVE",
        max_results: int = 25,
    ) -> list[PassageMatch]:
        """Asynchronously executes true SQL-level Hybrid Search with Reciprocal Rank Fusion (RRF, k=60).

        Combines:
        1. pgvector HNSW dense cosine similarity.
        2. PostgreSQL Full-Text Search (tsvector/tsquery) BM25 ranking (`ts_rank_cd`).
        3. Strict SQL WHERE pre-filtering on department, year range, status, and GO number.
        """
        if query_embedding is None:
            query_embedding = await self.a_generate_embedding(query_text)

        emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
        clean_query = DevanagariNormalizer.normalize_text(query_text.strip())

        # Extract year boundaries
        start_year: Optional[int] = None
        end_year: Optional[int] = None
        if year_range_filter:
            start_year = int(year_range_filter[0])
            end_year = int(year_range_filter[1]) if len(year_range_filter) > 1 else start_year

        # Normalize GO number filter
        clean_go_filter: Optional[str] = None
        if go_number_filter and go_number_filter.strip():
            raw_go = go_number_filter.strip().lstrip("GO-").strip()
            core_match = re.search(r"\b\d{2,5}\b", raw_go)
            clean_go_filter = f"%{core_match.group(0) if core_match else raw_go}%"

        # Expand department filter into bilingual keywords (e.g. Forest -> ['%वन%', '%पर्यावरण%', '%forest%'])
        dept_patterns: Optional[list[str]] = None
        if department_filter and department_filter.strip():
            keywords = get_dept_keywords(department_filter)
            dept_patterns = [f"%{kw}%" for kw in keywords]

        # SQL Hybrid Search with CTEs and Reciprocal Rank Fusion (RRF, k=60)
        sql_query = """
        WITH dense_matches AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector ASC) AS dense_rank,
                   (1.0 - (embedding <=> %s::vector)) AS dense_sim
            FROM document_chunks
            WHERE embedding IS NOT NULL
              AND (%s::text[] IS NULL OR issuing_department ILIKE ANY(%s::text[]))
              AND (%s::int IS NULL OR (year >= %s::int AND year <= %s::int))
              AND (%s::text IS NULL OR status = %s::text)
              AND (%s::text IS NULL OR go_number ILIKE %s::text)
            LIMIT 50
        ),
        sparse_matches AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv_content, plainto_tsquery('simple', %s)) DESC) AS sparse_rank,
                   ts_rank_cd(tsv_content, plainto_tsquery('simple', %s)) AS sparse_score
            FROM document_chunks
            WHERE (tsv_content @@ plainto_tsquery('simple', %s) OR exact_text_excerpt ILIKE %s)
              AND (%s::text[] IS NULL OR issuing_department ILIKE ANY(%s::text[]))
              AND (%s::int IS NULL OR (year >= %s::int AND year <= %s::int))
              AND (%s::text IS NULL OR status = %s::text)
              AND (%s::text IS NULL OR go_number ILIKE %s::text)
            LIMIT 50
        ),
        fused_ranks AS (
            SELECT
                COALESCE(d.id, s.id) AS chunk_id,
                (
                    COALESCE(1.0 / (60.0 + d.dense_rank), 0.0) +
                    COALESCE(1.0 / (60.0 + s.sparse_rank), 0.0)
                ) AS raw_rrf_score,
                COALESCE(d.dense_sim, 0.50) AS dense_sim
            FROM dense_matches d
            FULL OUTER JOIN sparse_matches s ON d.id = s.id
        )
        SELECT
            c.go_number,
            c.issuing_department,
            c.date,
            c.page_number,
            c.exact_text_excerpt,
            c.bounding_box_coordinates,
            f.raw_rrf_score,
            f.dense_sim
        FROM fused_ranks f
        JOIN document_chunks c ON f.chunk_id = c.id
        ORDER BY f.raw_rrf_score DESC
        LIMIT %s;
        """

        like_query_pattern = f"%{clean_query[:100]}%"
        params = [
            # Dense CTE
            emb_str,
            emb_str,
            dept_patterns,
            dept_patterns,
            start_year,
            start_year,
            end_year,
            status_filter,
            status_filter,
            clean_go_filter,
            clean_go_filter,
            # Sparse CTE
            clean_query,
            clean_query,
            clean_query,
            like_query_pattern,
            dept_patterns,
            dept_patterns,
            start_year,
            start_year,
            end_year,
            status_filter,
            status_filter,
            clean_go_filter,
            clean_go_filter,
            # Limit
            max_results,
        ]

        rows: list[dict[str, Any]] = []
        try:
            pool = await self.get_async_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql_query, params)
                    rows = await cur.fetchall()
        except Exception as exc:
            logger.warning("RRF SQL execution error (%s): %s; falling back to direct query", type(exc).__name__, exc)
            # Fallback to direct dense vector search
            try:
                fallback_sql = """
                SELECT go_number, issuing_department, date, page_number, exact_text_excerpt,
                       bounding_box_coordinates, (1.0 - (embedding <=> %s::vector)) AS dense_sim,
                       0.02 AS raw_rrf_score
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  AND (%s::text[] IS NULL OR issuing_department ILIKE ANY(%s::text[]))
                  AND (%s::text IS NULL OR go_number ILIKE %s::text)
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
                """
                pool = await self.get_async_pool()
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            fallback_sql,
                            [
                                emb_str,
                                dept_patterns,
                                dept_patterns,
                                clean_go_filter,
                                clean_go_filter,
                                emb_str,
                                max_results,
                            ],
                        )
                        rows = await cur.fetchall()
            except Exception as fb_exc:
                logger.error("Fallback vector query also failed: %s", fb_exc)
                return []

        results: list[PassageMatch] = []
        for r in rows:
            raw_bbox = r.get("bounding_box_coordinates")
            bbox: Optional[list[float]] = None
            if raw_bbox is not None:
                if isinstance(raw_bbox, list):
                    bbox = [float(v) for v in raw_bbox]
                elif isinstance(raw_bbox, str):
                    try:
                        parsed = json.loads(raw_bbox)
                        if isinstance(parsed, list):
                            bbox = [float(v) for v in parsed]
                    except Exception:
                        pass

            # Normalize score to [0.0, 1.0]
            # Max possible RRF score with k=60 across 2 ranks is (1/61 + 1/61) = 0.0328
            # We scale raw_rrf_score or blend with dense_sim for transparent downstream thresholding
            dense_sim = float(r.get("dense_sim") or 0.50)
            raw_rrf = float(r.get("raw_rrf_score") or 0.016)
            normalized_score = sanitize_relevance_score(max(dense_sim, min(1.0, raw_rrf * 30.0)))

            results.append(
                PassageMatch(
                    go_number=r["go_number"],
                    issuing_department=r["issuing_department"],
                    date=str(r["date"]),
                    page_number=r["page_number"],
                    exact_text_excerpt=r["exact_text_excerpt"],
                    relevance_score=normalized_score,
                    bounding_box_coordinates=bbox,
                )
            )

        # Fallback if filtered search returned nothing (Anti-Punting)
        if not results and department_filter:
            logger.warning(
                "Filtered search returned 0 results for dept '%s'. Retrying without department_filter...",
                department_filter,
            )
            return await self.a_hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                department_filter=None,
                year_range_filter=year_range_filter,
                go_number_filter=go_number_filter,
                status_filter=status_filter,
                max_results=max_results,
            )

        return results
