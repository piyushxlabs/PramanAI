"""Automated schema migration engine for ShasanAI PostgreSQL database.

Reads and applies SQL migration scripts in order, ensuring idempotent execution
for production RAG scaling with parent documents, supersession graph, and tsvector FTS.
"""

import logging
from pathlib import Path
import psycopg
from src.state.checkpointing import ensure_windows_event_loop

logger = logging.getLogger("shasanai.migrations")


def apply_migrations(db_url: str) -> None:
    """Applies all .sql migration files in src/ingestion/migrations to the database."""
    ensure_windows_event_loop()
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        logger.warning("Migrations directory not found: %s", migrations_dir)
        return

    sql_files = sorted(list(migrations_dir.glob("*.sql")))
    if not sql_files:
        logger.info("No migration SQL files found in %s", migrations_dir)
        return

    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Create migrations tracking table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute("SELECT filename FROM schema_migrations;")
            applied = {row[0] for row in cur.fetchall()}

            for sql_file in sql_files:
                if sql_file.name in applied:
                    continue

                logger.info("Applying database migration: %s", sql_file.name)
                sql_content = sql_file.read_text(encoding="utf-8")
                try:
                    cur.execute(sql_content)
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s);",
                        (sql_file.name,),
                    )
                    logger.info("Successfully applied migration: %s", sql_file.name)
                except Exception as exc:
                    logger.error("Failed to apply migration %s: %s", sql_file.name, exc)
                    raise


async def a_apply_migrations(db_url: str) -> None:
    """Asynchronously applies migrations in worker thread."""
    import asyncio
    await asyncio.to_thread(apply_migrations, db_url)
