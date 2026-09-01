"""PostgreSQL async checkpointing backend for ShasanAI StateGraph.

Utilizes AsyncPostgresSaver from langgraph-checkpoint-postgres with PostgreSQL 16 + pgvector.
Provides durable, multi-session persistence for 17-field StateSchema and HITL interrupt/resumption.
"""

import asyncio
from contextlib import asynccontextmanager
import os
import sys
import logging
from typing import Any, AsyncIterator
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool

from src.state.reducers import StateValidationError

load_dotenv()

logger = logging.getLogger("praman_ai.checkpointing")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shasanai"
)

_GLOBAL_MEMORY_SAVER = MemorySaver()


def ensure_windows_event_loop() -> None:
    """Ensures Windows uses SelectorEventLoop required by psycopg3 async."""
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass


@asynccontextmanager
async def get_checkpointer(
    database_url: str | None = None,
) -> AsyncIterator[Any]:
    """Async context manager yielding an AsyncPostgresSaver with automatic MemorySaver fallback.
    
    Args:
        database_url: PostgreSQL connection string. Defaults to DATABASE_URL env var.
        
    Yields:
        AsyncPostgresSaver | MemorySaver: Active checkpointer instance.
    """
    ensure_windows_event_loop()
    conn_str = database_url or DATABASE_URL

    if not conn_str:
        logger.warning("DATABASE_URL not specified; using in-memory checkpointer fallback.")
        yield _GLOBAL_MEMORY_SAVER
        return

    try:
        async with AsyncPostgresSaver.from_conn_string(conn_str) as saver:
            yield saver
    except Exception as exc:
        logger.warning(
            "AsyncPostgresSaver connection/event-loop issue (%s); falling back to resilient MemorySaver.",
            exc,
        )
        yield _GLOBAL_MEMORY_SAVER


async def setup_postgres_checkpoint_tables(
    database_url: str | None = None,
) -> None:
    """Initializes and migrates the required checkpointing tables in PostgreSQL.
    
    Creates: checkpoints, checkpoint_blobs, checkpoint_writes tables and indexes.
    """
    ensure_windows_event_loop()
    conn_str = database_url or DATABASE_URL

    async with AsyncPostgresSaver.from_conn_string(conn_str) as saver:
        await saver.setup()


setup_checkpoint_tables = setup_postgres_checkpoint_tables
