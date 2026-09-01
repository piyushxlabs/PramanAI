"""Sovereign JWT Authentication and Officer Access Control for ShasanAI.

Provides air-gapped JWT token creation, bcrypt password validation,
and FastAPI dependency injection for multi-tenant officer roles.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

from src.server.schemas import UserProfile
from src.state.checkpointing import ensure_windows_event_loop

logger = logging.getLogger("shasanai.auth")
load_dotenv()
ensure_windows_event_loop()

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------

JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "shasanai_sovereign_secret_key_uk_2026_itda_secure_airgap"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24-hour persistent officer session

security_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Password Management
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hashes a password using bcrypt with 12 rounds of salt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception as exc:
        logger.warning(f"Password verification error: {exc}")
        return False


# ---------------------------------------------------------------------------
# JWT Token Generation & Verification
# ---------------------------------------------------------------------------

def create_access_token(
    claims: dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Encodes claims into a sovereign JWT access token."""
    to_encode = claims.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodes and verifies a JWT token. Raises HTTPException on expiration or tampering."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc!s}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Database User Retrieval
# ---------------------------------------------------------------------------

def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Retrieves user record from PostgreSQL users table by email."""
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, hashed_password, full_name, department, designation, role, created_at
                FROM users
                WHERE email = %s
                LIMIT 1;
                """,
                (email.strip().lower(),),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    """Retrieves user record from PostgreSQL users table by primary key ID."""
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, hashed_password, full_name, department, designation, role, created_at
                FROM users
                WHERE id = %s
                LIMIT 1;
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    return None


# ---------------------------------------------------------------------------
# FastAPI Dependency Injection
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> UserProfile:
    """Validates JWT Bearer authorization and returns the active UserProfile.
    
    Raises HTTP 401 if missing, invalid, or user not found.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing subject identifier ('sub').",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(sub)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid subject identifier in token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_data = get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserProfile(
        id=user_data["id"],
        email=user_data["email"],
        full_name=user_data["full_name"],
        department=user_data["department"],
        designation=user_data["designation"],
        role=user_data.get("role", "OFFICER"),
        created_at=user_data["created_at"].isoformat() if user_data.get("created_at") else None,
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[UserProfile]:
    """Extracts UserProfile if valid Bearer token provided, otherwise returns None."""
    if not credentials or not credentials.credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
