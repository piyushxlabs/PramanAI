"""Pytest configuration and Windows asyncio loop fixtures for PostgreSQL async driver compatibility."""

import asyncio
import selectors
import sys
import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="function")
def event_loop():
    """Provides a SelectorEventLoop on Windows for psycopg3 async compatibility."""
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    else:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
