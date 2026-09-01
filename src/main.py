"""Entrypoint launcher for PramanAI FastAPI Backend Server."""
import sys
import asyncio
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_server() -> None:
    """Runs the FastAPI server with Windows-compatible SelectorEventLoop."""
    host = os.getenv("API_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))

    if sys.platform == "win32":
        # Force SelectorEventLoop for Uvicorn on Windows
        loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            "src.server.app:app",
            host=host,
            port=port,
            reload=False,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())
    else:
        uvicorn.run("src.server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()

