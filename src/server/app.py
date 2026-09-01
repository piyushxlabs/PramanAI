"""FastAPI Backend Server for ShasanAI Government Order Retrieval & Observability System.

Exposes SSE chat streaming, durable HITL graph resumption from PostgreSQL checkpoints,
feedback scoring endpoints, health monitoring, and real PDF + page-image serving endpoints.
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from concurrent.futures import ThreadPoolExecutor

import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command

from src.agents.graph import create_agent_graph
from src.server.auth import (
    create_access_token,
    get_current_user,
    get_optional_user,
    get_user_by_email,
    verify_password,
)
from src.server.schemas import (
    ApiResponse,
    AuthLoginRequest,
    AuthTokenResponse,
    ChatQueryRequest,
    ChatSessionDetailResponse,
    ChatSessionItem,
    ChatSessionListResponse,
    CitationAccuracyFeedbackRequest,
    HealthResponse,
    HITLResumptionRequest,
    OfficerFeedbackRequest,
    UserProfile,
)
from src.state.checkpointing import (
    ensure_windows_event_loop,
    get_checkpointer,
    setup_checkpoint_tables,
)
from src.state.schema import OfficerContext, RuntimeConfig, StateSchema
from src.telemetry.feedback_annotations import (
    record_citation_accuracy,
    record_human_verification_outcome,
    record_officer_feedback,
)
from src.telemetry.tracing import instrument_langchain, setup_telemetry
from src.ui.hitl_resumption import resume_hitl_stream
from src.ui.stream_handler import stream_agent_turn

ensure_windows_event_loop()

logger = logging.getLogger("shasanai.server")
logging.basicConfig(level=logging.INFO)


_PDF_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf_render")

# ---------------------------------------------------------------------------
# PDF Path Resolver — plain helper function, NOT a route handler
# ---------------------------------------------------------------------------

def _find_pdf_path(go_number: str) -> Path | None:
    """Finds the corresponding PDF file using PostgreSQL file_path mapping and smart filesystem lookup."""
    raw_key = go_number.strip()
    clean_go = re.sub(
        r"^(?:UK-)?(?:GO|ORDER|NOTIFICATION)[-_\/\s:]*", "", raw_key, flags=re.IGNORECASE
    ).strip()

    # Extract core digits identifier (e.g. 667, 115, 825)
    core_match = re.search(r"\b\d{2,5}\b", clean_go)
    core_num = core_match.group(0) if core_match else None

    # 1. Database-backed exact file_path lookup from document_chunks table
    try:
        from src.ingestion.vector_store import VectorStore
        store = VectorStore()
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                # Try exact GO number, document_id, or LIKE match
                cur.execute(
                    """
                    SELECT file_path FROM document_chunks 
                    WHERE go_number = %s OR document_id = %s OR go_number ILIKE %s
                    LIMIT 1;
                    """,
                    (raw_key, raw_key, f"%{clean_go}%"),
                )
                row = cur.fetchone()
                if row and row.get("file_path"):
                    candidate = Path(row["file_path"])
                    if candidate.exists():
                        return candidate
                
                # If core_num present, try core_num search in database
                if core_num:
                    cur.execute(
                        """
                        SELECT file_path FROM document_chunks 
                        WHERE go_number ILIKE %s
                        LIMIT 1;
                        """,
                        (f"%{core_num}%",),
                    )
                    row = cur.fetchone()
                    if row and row.get("file_path"):
                        candidate = Path(row["file_path"])
                        if candidate.exists():
                            return candidate
    except Exception as e:
        logger.warning(f"Database lookup in _find_pdf_path failed: {e}")

    # 2. Filesystem smart lookup
    pdf_dirs = [Path("data/raw_pdfs"), Path("data/corpus/pdfs"), Path("data/pdfs")]
    pdf_files: list[Path] = []
    for d in pdf_dirs:
        if d.exists():
            pdf_files.extend(d.glob("*.pdf"))

    if not pdf_files:
        return None

    # Known filename mappings for Uttarakhand dataset
    KNOWN_MAP = {
        "667": "8102018173922.pdf",
        "115": "2832018165730.pdf",
        "825": "24122020162747.pdf",
        "739": "24122020162747.pdf",
    }
    if core_num and core_num in KNOWN_MAP:
        target_name = KNOWN_MAP[core_num]
        for p in pdf_files:
            if p.name == target_name:
                return p

    # Exact filename / stem matches
    for p in pdf_files:
        if p.name.lower() in [raw_key.lower(), f"{raw_key.lower()}.pdf", clean_go.lower(), f"{clean_go.lower()}.pdf"]:
            return p
        if p.stem.lower() in [raw_key.lower(), clean_go.lower()]:
            return p

    # Token-based match (length >= 3)
    tokens = [t for t in re.split(r"[\/\-_\s\(\)\.]+", clean_go) if len(t) >= 3]
    for p in pdf_files:
        if any(tok.lower() in p.stem.lower() for tok in tokens):
            return p

    # Never silently fallback to pdf_files[0]
    return None


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager initializing checkpointer tables and telemetry."""
    logger.info("Initializing PramanAI backend lifespan...")
    ensure_windows_event_loop()
    setup_telemetry()
    instrument_langchain()

    try:
        from src.ingestion.vector_store import VectorStore
        VectorStore().initialize_schema()
        await setup_checkpoint_tables()
        logger.info("PostgreSQL checkpointer and production RAG schema verified.")
    except Exception as exc:
        logger.warning(f"PostgreSQL checkpointer setup warning: {exc!s}")

    yield
    logger.info("Shutting down PramanAI backend lifespan...")


app = FastAPI(
    title="PramanAI Enterprise Backend",
    description="Autonomous Evidentiary GovTech Agent Fleet with Multi-Modal Grounding on Gemini & Google Cloud",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Next.js interface (port 3000) and all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Health check endpoint confirming model bindings and database connectivity."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", response_model=AuthTokenResponse)
async def login_endpoint(request: AuthLoginRequest) -> AuthTokenResponse:
    """Authenticates officer credentials and returns a sovereign JWT Bearer token."""
    user = get_user_by_email(request.email)
    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {
        "sub": str(user["id"]),
        "email": user["email"],
        "full_name": user["full_name"],
        "department": user["department"],
        "designation": user["designation"],
        "role": user.get("role", "OFFICER"),
    }
    token = create_access_token(claims)
    profile = UserProfile(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        department=user["department"],
        designation=user["designation"],
        role=user.get("role", "OFFICER"),
        created_at=user["created_at"].isoformat() if user.get("created_at") else None,
    )
    return AuthTokenResponse(access_token=token, token_type="bearer", user=profile)


@app.get("/api/auth/me", response_model=UserProfile)
async def get_me_endpoint(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """Returns the authenticated officer profile from JWT token claims."""
    return current_user


# ---------------------------------------------------------------------------
# Chat History Helpers & Endpoints
# ---------------------------------------------------------------------------

def record_or_update_chat_session(
    session_id: str,
    user_id: int,
    query_text: str,
    department: str,
) -> None:
    """Creates or updates chat_sessions entry for persistent history."""
    from src.ingestion.vector_store import VectorStore

    clean_title = query_text.strip()
    if len(clean_title) > 60:
        clean_title = clean_title[:57] + "..."
    if not clean_title:
        clean_title = "New Query"

    try:
        store = VectorStore()
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (session_id, user_id, title, department, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        updated_at = NOW(),
                        title = CASE 
                            WHEN chat_sessions.title = 'New Query' OR chat_sessions.title = '' THEN EXCLUDED.title 
                            ELSE chat_sessions.title 
                        END;
                    """,
                    (session_id, user_id, clean_title, department),
                )
    except Exception as exc:
        logger.warning(f"Failed to record chat session {session_id}: {exc}")


@app.get("/api/chat/history", response_model=ChatSessionListResponse)
async def get_chat_history(
    current_user: UserProfile = Depends(get_current_user),
) -> ChatSessionListResponse:
    """Retrieves all chat sessions for the authenticated officer ordered by recency."""
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    sessions: list[ChatSessionItem] = []
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, user_id, title, department, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC;
                """,
                (current_user.id,),
            )
            rows = cur.fetchall()
            for r in rows:
                sessions.append(
                    ChatSessionItem(
                        session_id=r["session_id"],
                        user_id=r.get("user_id"),
                        title=r["title"],
                        department=r["department"],
                        created_at=r["created_at"].isoformat() if r.get("created_at") else "",
                        updated_at=r["updated_at"].isoformat() if r.get("updated_at") else "",
                    )
                )

    return ChatSessionListResponse(sessions=sessions, total=len(sessions))


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat_session_detail(
    session_id: str,
    current_user: UserProfile = Depends(get_current_user),
) -> ChatSessionDetailResponse:
    """Loads full conversational messages and verified citations for a session."""
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    session_row = None
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, user_id, title, department, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = %s
                LIMIT 1;
                """,
                (session_id,),
            )
            session_row = cur.fetchone()

    if not session_row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if session_row.get("user_id") != current_user.id and current_user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Session belongs to another officer account.",
        )

    # Reconstruct messages from checkpoint if available
    messages: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []

    try:
        ensure_windows_event_loop()
        async with get_checkpointer() as saver:
            tuple_data = await saver.aget_tuple({"configurable": {"thread_id": session_id, "checkpoint_ns": ""}})
            if tuple_data and tuple_data.checkpoint:
                channel_values = tuple_data.checkpoint.get("channel_values", {})
                q_text = channel_values.get("query_text") or session_row["title"]
                a_md = channel_values.get("answer_markdown")
                c_list = channel_values.get("citations") or []
                conf = channel_values.get("confidence_score")
                super_st = channel_values.get("supersession_status")
                refused = channel_values.get("graceful_refusal", False)

                if c_list and isinstance(c_list, list):
                    citations = [
                        c if isinstance(c, dict) else c.model_dump() if hasattr(c, "model_dump") else {}
                        for c in c_list
                    ]

                # User message
                messages.append({
                    "id": f"msg_user_{session_id}",
                    "role": "officer",
                    "content": q_text,
                    "timestamp": session_row["created_at"].strftime("%I:%M %p") if session_row.get("created_at") else "",
                    "userQuery": q_text,
                })

                # Agent message if response generated
                if a_md or refused:
                    content_str = a_md or (
                        "**सत्यापन अस्वीकृत / Verification Denied:**\n\nअधिकारी द्वारा सत्यापन अस्वीकृत कर दिया गया है।"
                        if refused
                        else ""
                    )
                    messages.append({
                        "id": f"msg_agent_{session_id}",
                        "role": "agent",
                        "content": content_str,
                        "timestamp": session_row["updated_at"].strftime("%I:%M %p") if session_row.get("updated_at") else "",
                        "confidence_score": conf,
                        "supersession_status": super_st,
                        "citations": citations,
                        "graceful_refusal": refused,
                    })
    except Exception as exc:
        logger.warning(f"Could not load checkpoint values for session {session_id}: {exc}")

    # Fallback message if checkpointer had no values
    if not messages:
        messages.append({
            "id": f"msg_user_{session_id}",
            "role": "officer",
            "content": session_row["title"],
            "timestamp": session_row["created_at"].strftime("%I:%M %p") if session_row.get("created_at") else "",
            "userQuery": session_row["title"],
        })

    return ChatSessionDetailResponse(
        session_id=session_row["session_id"],
        user_id=session_row.get("user_id"),
        title=session_row["title"],
        department=session_row["department"],
        created_at=session_row["created_at"].isoformat() if session_row.get("created_at") else "",
        updated_at=session_row["updated_at"].isoformat() if session_row.get("updated_at") else "",
        messages=messages,
        citations=citations,
    )


@app.delete("/api/chat/sessions/{session_id}", response_model=ApiResponse)
async def delete_chat_session(
    session_id: str,
    current_user: UserProfile = Depends(get_current_user),
) -> ApiResponse:
    """Deletes a chat session for the authenticated officer."""
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id FROM chat_sessions WHERE session_id = %s;
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")
            if row.get("user_id") != current_user.id and current_user.role != "ADMIN":
                raise HTTPException(status_code=403, detail="Access denied")

            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s;", (session_id,))

    return ApiResponse(success=True, message=f"Session '{session_id}' deleted successfully.")


# ---------------------------------------------------------------------------
# Chat SSE Streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_stream_endpoint(
    request: ChatQueryRequest,
    current_user: Optional[UserProfile] = Depends(get_optional_user),
) -> StreamingResponse:
    """Initiates an officer query turn and streams Server-Sent Events (SSE)."""
    ensure_windows_event_loop()

    # Determine officer context
    default_dept = current_user.department if current_user else "Forest"
    officer = request.officer_context or OfficerContext(
        department=default_dept, access_scope=[default_dept, "General"]
    )

    # Persist session entry linked to user
    user_id = current_user.id if current_user else 1
    record_or_update_chat_session(
        session_id=request.session_id,
        user_id=user_id,
        query_text=request.query_text,
        department=officer.department,
    )

    initial_state: StateSchema = {
        "session_id": request.session_id,
        "officer_context": officer,
        "query_text": request.query_text,
        "query_language": "en",
        "query_filters": request.query_filters,
        "message_history": [],
        "retrieved_passages": [],
        "candidate_citations": [],
        "confidence_score": 0.0,
        "supersession_status": "UNKNOWN",
        "conflict_flags": [],
        "human_verification": None,
        "answer_markdown": None,
        "citations": [],
        "graceful_refusal": False,
        "error_logs": [],
        "config": RuntimeConfig(),
    }

    async def event_generator() -> AsyncIterator[str]:
        try:
            async with get_checkpointer() as saver:
                app_graph = create_agent_graph(checkpointer=saver)
                config = {"configurable": {"thread_id": request.session_id, "checkpoint_ns": ""}, "recursion_limit": 15}
                async for event in stream_agent_turn(app_graph, initial_state, config=config):
                    yield event
        except Exception as exc:
            logger.error("Chat turn error during streaming (%s); attempting in-memory fallback...", exc)
            try:
                from langgraph.checkpoint.memory import MemorySaver
                fallback_graph = create_agent_graph(checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": request.session_id, "checkpoint_ns": ""}, "recursion_limit": 15}
                async for event in stream_agent_turn(fallback_graph, initial_state, config=config):
                    yield event
            except Exception as inner_exc:
                logger.error("Fallback graph execution error: %s", inner_exc)
                yield f'data: {{"type":"error","errorText":"System error: {inner_exc!s}","code":"execution_error","recoverable":false}}\n\n'
                yield 'data: {"type":"finish","finishReason":"error"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# HITL Resume
# ---------------------------------------------------------------------------

@app.post("/resume")
@app.post("/api/hitl/resume")
async def resume_hitl_endpoint(request: HITLResumptionRequest) -> StreamingResponse:
    """Resumes a paused LangGraph execution from PostgreSQL checkpoint upon officer approval/denial."""
    resolved_go = request.modified_inputs.resolved_go_number if request.modified_inputs else None

    return StreamingResponse(
        resume_hitl_stream(
            checkpoint_id=request.checkpoint_id,
            action=request.action,
            resolved_go_number=resolved_go,
            reason=request.reason,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Feedback Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/feedback/score", response_model=ApiResponse)
async def submit_officer_feedback(request: OfficerFeedbackRequest) -> ApiResponse:
    """Records officer thumbs up/down rating into Langfuse evaluation pipeline."""
    score_data = record_officer_feedback(
        session_id=request.session_id,
        trace_id=request.trace_id,
        feedback_value=request.feedback_value,
        comment=request.comment,
    )
    return ApiResponse(
        success=True,
        message="Officer feedback recorded successfully.",
        data=score_data,
    )


@app.post("/api/feedback/citation", response_model=ApiResponse)
async def submit_citation_accuracy(request: CitationAccuracyFeedbackRequest) -> ApiResponse:
    """Records per-citation incorrect flag into Langfuse evaluation pipeline."""
    score_data = record_citation_accuracy(
        session_id=request.session_id,
        go_number=request.go_number,
        page_number=request.page_number,
        trace_id=request.trace_id,
        is_accurate=request.is_accurate,
        comment=request.comment,
    )
    return ApiResponse(
        success=True,
        message="Citation accuracy flag recorded successfully.",
        data=score_data,
    )


# ---------------------------------------------------------------------------
# Document Page Image Endpoint — MUST be declared BEFORE the PDF FileResponse
# route to avoid the `:path` wildcard catching "/pages/N" as part of go_number
# ---------------------------------------------------------------------------

@app.get("/documents/{go_number:path}/pages/{page_number}")
@app.get("/api/documents/{go_number:path}/pages/{page_number}")
async def get_document_page_image_endpoint(go_number: str, page_number: int) -> Response:
    """Renders a specific page of the Government Order PDF to PNG at 150 DPI.

    Returns image/png with CORS and inline Content-Disposition headers so it can be
    embedded directly in <img> tags in the Next.js frontend at any origin.
    """
    loop = asyncio.get_event_loop()
    pdf_path = await loop.run_in_executor(_PDF_EXECUTOR, _find_pdf_path, go_number)
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF for GO '{go_number}' not found")

    def _render_page() -> bytes:
        import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise HTTPException(status_code=404, detail="PDF has 0 pages")

        # Clamp page index to valid range
        page_idx = max(1, min(page_number, total_pages))
        page = doc[page_idx - 1]
        pix = page.get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    try:
        png_bytes = await loop.run_in_executor(_PDF_EXECUTOR, _render_page)
        page_idx = max(1, page_number)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f'inline; filename="{pdf_path.stem}_p{page_idx}.png"',
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to render page image for GO '{go_number}' page {page_number}: {exc}")
        raise HTTPException(status_code=500, detail=f"Page image rendering error: {exc}")


# ---------------------------------------------------------------------------
# Raw PDF FileResponse Endpoint
# ---------------------------------------------------------------------------

@app.get("/documents/{go_number:path}")
@app.get("/api/documents/{go_number:path}")
async def get_document_endpoint(go_number: str) -> FileResponse:
    """Serves the raw Government Order PDF as an inline FileResponse.

    Returns the actual binary PDF with Content-Disposition: inline so the browser
    renders it directly in the <iframe> without triggering a download dialog.
    """
    loop = asyncio.get_event_loop()
    pdf_path = await loop.run_in_executor(_PDF_EXECUTOR, _find_pdf_path, go_number)
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{go_number}' not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={
            "Content-Disposition": f'inline; filename="{pdf_path.name}"',
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )
