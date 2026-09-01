"""Verification Suite for Sovereign JWT Authentication and Isolated Chat History.

Validates:
1. Successful login and JWT bearer token issuance for all 4 seeded officer personas.
2. Rejection of invalid credentials (wrong password and non-existent email).
3. JWT claim integrity and user profile extraction in /api/auth/me.
4. Strict multi-tenant session isolation across different officer user_ids.
5. Session detail retrieval and session deletion capabilities.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.ingestion.vector_store import VectorStore
from src.server.app import app
from src.state.checkpointing import ensure_windows_event_loop

ensure_windows_event_loop()

# Seeded accounts to verify
OFFICER_ACCOUNTS = [
    ("forest.officer@uk.gov.in", "Shasan@2026", "Forest", "Vikram Singh Negi", "OFFICER"),
    ("finance.officer@uk.gov.in", "Shasan@2026", "Finance", "Pooja Sharma", "OFFICER"),
    ("personnel.officer@uk.gov.in", "Shasan@2026", "Personnel", "Rajesh Chandra", "OFFICER"),
    ("admin.itda@uk.gov.in", "Shasan@2026", "General", "Amitabh Rawat", "ADMIN"),
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provides FastAPI test client and ensures database schema is initialized."""
    VectorStore().initialize_schema()
    return TestClient(app)


def test_seeded_accounts_login_success(client: TestClient) -> None:
    """Verifies that all 4 pre-seeded Uttarakhand Secretariat personas can authenticate successfully."""
    for email, password, dept, full_name, role in OFFICER_ACCOUNTS:
        response = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        assert response.status_code == status.HTTP_200_OK, f"Failed login for {email}: {response.text}"
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == email
        assert data["user"]["department"] == dept
        assert data["user"]["full_name"] == full_name
        assert data["user"]["role"] == role


def test_invalid_credentials_rejected(client: TestClient) -> None:
    """Verifies that wrong password and non-existent accounts are rejected with 401 Unauthorized."""
    # Wrong password
    resp1 = client.post(
        "/api/auth/login",
        json={"email": "forest.officer@uk.gov.in", "password": "WrongPassword123!"},
    )
    assert resp1.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid email or password" in resp1.json()["detail"]

    # Non-existent user
    resp2 = client.post(
        "/api/auth/login",
        json={"email": "nonexistent.officer@uk.gov.in", "password": "Shasan@2026"},
    )
    assert resp2.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid email or password" in resp2.json()["detail"]


def test_jwt_claim_extraction_me_endpoint(client: TestClient) -> None:
    """Verifies that /api/auth/me accurately extracts and verifies JWT claims."""
    # 1. Login as Finance Officer
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "finance.officer@uk.gov.in", "password": "Shasan@2026"},
    )
    assert login_resp.status_code == status.HTTP_200_OK
    token = login_resp.json()["access_token"]

    # 2. Call /api/auth/me with valid Bearer token
    me_resp = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == status.HTTP_200_OK
    profile = me_resp.json()
    assert profile["email"] == "finance.officer@uk.gov.in"
    assert profile["full_name"] == "Pooja Sharma"
    assert profile["department"] == "Finance"
    assert profile["role"] == "OFFICER"

    # 3. Call /api/auth/me without token -> 401
    unauth_resp = client.get("/api/auth/me")
    assert unauth_resp.status_code == status.HTTP_401_UNAUTHORIZED

    # 4. Call /api/auth/me with malformed token -> 401
    bad_token_resp = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid_garbage_token_123"},
    )
    assert bad_token_resp.status_code == status.HTTP_401_UNAUTHORIZED


def test_chat_history_strict_user_isolation(client: TestClient) -> None:
    """Verifies strict multi-tenant isolation: Officer A cannot view or access Officer B's sessions."""
    # Log in as Forest Officer
    forest_login = client.post(
        "/api/auth/login",
        json={"email": "forest.officer@uk.gov.in", "password": "Shasan@2026"},
    )
    forest_token = forest_login.json()["access_token"]
    forest_id = forest_login.json()["user"]["id"]

    # Log in as Finance Officer
    finance_login = client.post(
        "/api/auth/login",
        json={"email": "finance.officer@uk.gov.in", "password": "Shasan@2026"},
    )
    finance_token = finance_login.json()["access_token"]
    finance_id = finance_login.json()["user"]["id"]

    # Insert a test session for Forest Officer directly into DB
    forest_session_id = "test_forest_session_xyz_1001"
    store = VectorStore()
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, title, department, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title;
                """,
                (forest_session_id, forest_id, "Forest Conservation Tree Felling Rules 2018", "Forest"),
            )

    # 1. Forest Officer should see this session in their history
    forest_hist = client.get(
        "/api/chat/history",
        headers={"Authorization": f"Bearer {forest_token}"},
    )
    assert forest_hist.status_code == status.HTTP_200_OK
    forest_sessions = forest_hist.json()["sessions"]
    forest_ids = [s["session_id"] for s in forest_sessions]
    assert forest_session_id in forest_ids

    # 2. Finance Officer MUST NOT see Forest Officer's session in their history
    finance_hist = client.get(
        "/api/chat/history",
        headers={"Authorization": f"Bearer {finance_token}"},
    )
    assert finance_hist.status_code == status.HTTP_200_OK
    finance_sessions = finance_hist.json()["sessions"]
    finance_ids = [s["session_id"] for s in finance_sessions]
    assert forest_session_id not in finance_ids, "Security breach: Finance Officer can see Forest Officer session!"

    # 3. Finance Officer trying to access Forest Officer's session detail directly -> 403 Forbidden
    detail_resp = client.get(
        f"/api/chat/sessions/{forest_session_id}",
        headers={"Authorization": f"Bearer {finance_token}"},
    )
    assert detail_resp.status_code == status.HTTP_403_FORBIDDEN

    # 4. Forest Officer accessing their own session detail -> 200 OK
    forest_detail_resp = client.get(
        f"/api/chat/sessions/{forest_session_id}",
        headers={"Authorization": f"Bearer {forest_token}"},
    )
    assert forest_detail_resp.status_code == status.HTTP_200_OK
    detail_data = forest_detail_resp.json()
    assert detail_data["session_id"] == forest_session_id
    assert detail_data["title"] == "Forest Conservation Tree Felling Rules 2018"
    assert detail_data["department"] == "Forest"

    # 5. Clean up test session via DELETE endpoint
    del_resp = client.delete(
        f"/api/chat/sessions/{forest_session_id}",
        headers={"Authorization": f"Bearer {forest_token}"},
    )
    assert del_resp.status_code == status.HTTP_200_OK
    assert del_resp.json()["success"] is True


def test_admin_access_override(client: TestClient) -> None:
    """Verifies that ITDA Admin (ADMIN role) can inspect sessions across departments for governance/audit."""
    # Login as Personnel Officer
    personnel_login = client.post(
        "/api/auth/login",
        json={"email": "personnel.officer@uk.gov.in", "password": "Shasan@2026"},
    )
    personnel_token = personnel_login.json()["access_token"]
    personnel_id = personnel_login.json()["user"]["id"]

    # Login as ITDA Admin
    admin_login = client.post(
        "/api/auth/login",
        json={"email": "admin.itda@uk.gov.in", "password": "Shasan@2026"},
    )
    admin_token = admin_login.json()["access_token"]

    personnel_session_id = "test_personnel_session_audit_2002"
    store = VectorStore()
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, title, department, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title;
                """,
                (personnel_session_id, personnel_id, "Grade Pay Revision Order 2020", "Personnel"),
            )

    # Admin accesses Personnel Officer session detail -> 200 OK (Auditor/Admin governance privilege)
    admin_detail_resp = client.get(
        f"/api/chat/sessions/{personnel_session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_detail_resp.status_code == status.HTTP_200_OK
    assert admin_detail_resp.json()["session_id"] == personnel_session_id

    # Clean up
    client.delete(
        f"/api/chat/sessions/{personnel_session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
