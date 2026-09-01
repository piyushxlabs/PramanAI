"""Exhaustive unit test suite for Step 3: StateSchema, Reducers, and Pydantic V2 Models.

Tests all 17 fields, all 5 reducer types, exception hierarchy, and validation boundaries.
"""

import pytest
from pydantic import ValidationError

from src.state.reducers import (
    AgentError,
    ApprovalTimeoutError,
    ScopeViolationError,
    StateValidationError,
    ToolExecutionError,
    append_only_reducer,
    immutable_reducer,
    last_write_wins_reducer,
    merge_by_citation_key_reducer,
    replace_on_new_turn_reducer,
)
from src.state.schema import (
    ApprovalState,
    Citation,
    CitationIntegrityResult,
    CompareGoVersionsInput,
    CompareGoVersionsOutput,
    ConfidenceSupersessionAssessment,
    ConflictRecord,
    ErrorRecord,
    GetSourceHighlightInput,
    GetSourceHighlightOutput,
    GroundedAnswer,
    Message,
    OfficerContext,
    PassageMatch,
    QueryFilters,
    QueryInterpretation,
    RuntimeConfig,
    ScopeScreenDecision,
    SearchGoCorpusInput,
    SearchGoCorpusOutput,
    StateSchema,
    SupersessionLink,
)


# ===========================================================================
# 1. Custom AgentError Hierarchy Tests
# ===========================================================================

def test_exception_hierarchy():
    """Verify custom exception hierarchy rooted at AgentError."""
    assert issubclass(StateValidationError, AgentError)
    assert issubclass(ToolExecutionError, AgentError)
    assert issubclass(ApprovalTimeoutError, AgentError)
    assert issubclass(ScopeViolationError, AgentError)
    assert issubclass(AgentError, Exception)


# ===========================================================================
# 2. Immutable-After-Init Reducer Tests (session_id, officer_context, config)
# ===========================================================================

def test_immutable_session_id():
    """Test session_id immutability."""
    # Init
    state = immutable_reducer(None, "sess_001")
    assert state == "sess_001"

    # Same value re-write permitted
    state = immutable_reducer(state, "sess_001")
    assert state == "sess_001"

    # Mutation raises StateValidationError
    with pytest.raises(StateValidationError):
        immutable_reducer(state, "sess_002")


def test_immutable_officer_context():
    """Test officer_context immutability."""
    ctx1 = OfficerContext(department="Forest", access_scope=["Forest", "General"])
    ctx2 = OfficerContext(department="Revenue", access_scope=["Revenue"])

    # Init
    state = immutable_reducer(None, ctx1)
    assert state == ctx1

    # Same value permitted
    state = immutable_reducer(state, ctx1)
    assert state.department == "Forest"

    # Mutation to different department/scope raises StateValidationError
    with pytest.raises(StateValidationError):
        immutable_reducer(state, ctx2)


def test_immutable_runtime_config():
    """Test runtime config immutability."""
    cfg1 = RuntimeConfig(confidence_threshold_low=0.6, confidence_threshold_high=0.85)
    cfg2 = RuntimeConfig(confidence_threshold_low=0.4, confidence_threshold_high=0.90)

    state = immutable_reducer(None, cfg1)
    assert state == cfg1

    state = immutable_reducer(state, cfg1)
    assert state.confidence_threshold_low == 0.6

    with pytest.raises(StateValidationError):
        immutable_reducer(state, cfg2)


# ===========================================================================
# 3. Append-Only Reducer Tests (message_history, conflict_flags, error_logs)
# ===========================================================================

def test_append_only_message_history():
    """Test message_history non-destructive accumulation."""
    m1 = Message(role="user", content="Query 1")
    m2 = Message(role="assistant", content="Answer 1")
    m3 = Message(role="user", content="Follow-up Query 2")

    # Initial append
    history = append_only_reducer([], [m1])
    assert len(history) == 1
    assert history[0] == m1

    # Appending second turn
    history = append_only_reducer(history, [m2])
    assert len(history) == 2
    assert history[0] == m1
    assert history[1] == m2

    # Appending single item (not list)
    history = append_only_reducer(history, m3)
    assert len(history) == 3
    assert history[2] == m3


def test_append_only_conflict_flags():
    """Test conflict_flags accumulation per turn."""
    c1 = ConflictRecord(go_numbers=["GO-101", "GO-102"], description="Conflicting retirement age")
    c2 = ConflictRecord(go_numbers=["GO-201", "GO-202"], description="Conflicting pay scale")

    flags = append_only_reducer([], [c1])
    assert len(flags) == 1

    flags = append_only_reducer(flags, [c2])
    assert len(flags) == 2
    assert flags[0].go_numbers == ["GO-101", "GO-102"]
    assert flags[1].go_numbers == ["GO-201", "GO-202"]


def test_append_only_error_logs():
    """Test error_logs audit logging preservation."""
    e1 = ErrorRecord(node="scope_screen", error_type="out_of_scope", message="Grievance request redirected")
    e2 = ErrorRecord(node="citation_integrity", error_type="uncited_claim", message="Claim retry triggered")

    logs = append_only_reducer([], [e1])
    assert len(logs) == 1

    logs = append_only_reducer(logs, [e2])
    assert len(logs) == 2
    assert logs[0].node == "scope_screen"
    assert logs[1].node == "citation_integrity"


# ===========================================================================
# 4. Merge-by-Key Reducer Tests (candidate_citations)
# ===========================================================================

def test_merge_by_citation_key():
    """Test candidate_citations deduplication on (go_number, page_number)."""
    cit_a1 = Citation(
        go_number="GO-100/2018",
        issuing_department="Forest",
        date="2018-05-12",
        page_number=3,
        exact_text_excerpt="First retrieval excerpt",
    )
    cit_a2 = Citation(
        go_number="GO-100/2018",
        issuing_department="Forest",
        date="2018-05-12",
        page_number=3,
        exact_text_excerpt="Updated sparse retrieval excerpt",
        bounding_box_coordinates={"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0},
    )
    cit_b = Citation(
        go_number="GO-200/2019",
        issuing_department="Finance",
        date="2019-08-20",
        page_number=1,
        exact_text_excerpt="Finance excerpt",
    )

    # Initial merge
    merged = merge_by_citation_key_reducer([], [cit_a1])
    assert len(merged) == 1

    # Overwrite same (go_number, page_number)
    merged = merge_by_citation_key_reducer(merged, [cit_a2])
    assert len(merged) == 1
    assert merged[0].exact_text_excerpt == "Updated sparse retrieval excerpt"
    assert merged[0].bounding_box_coordinates is not None

    # Add distinct GO
    merged = merge_by_citation_key_reducer(merged, [cit_b])
    assert len(merged) == 2
    assert {c.go_number for c in merged} == {"GO-100/2018", "GO-200/2019"}


def test_merge_by_citation_key_with_dicts():
    """Test candidate_citations merge with dictionary payloads."""
    d1 = {"go_number": "GO-100", "page_number": 1, "exact_text_excerpt": "Text 1"}
    d2 = {"go_number": "GO-100", "page_number": 1, "exact_text_excerpt": "Text 1 Updated"}
    d3 = {"go_number": "GO-100", "page_number": 2, "exact_text_excerpt": "Text Page 2"}

    merged = merge_by_citation_key_reducer([d1], [d2, d3])
    assert len(merged) == 2
    assert merged[0]["exact_text_excerpt"] == "Text 1 Updated"
    assert merged[1]["page_number"] == 2


# ===========================================================================
# 5. Replace-on-New-Turn Reducer Tests (retrieved_passages)
# ===========================================================================

def test_replace_on_new_turn_passages():
    """Test retrieved_passages cleanly replaces working retrieval state on new turn."""
    p1 = PassageMatch(
        go_number="GO-1", issuing_department="Dept", date="2020-01-01", page_number=1, exact_text_excerpt="Turn 1 text", relevance_score=0.9
    )
    p2 = PassageMatch(
        go_number="GO-2", issuing_department="Dept", date="2021-01-01", page_number=1, exact_text_excerpt="Turn 2 text", relevance_score=0.95
    )

    # Turn 1
    state = replace_on_new_turn_reducer([], [p1])
    assert len(state) == 1
    assert state[0].go_number == "GO-1"

    # Turn 2 cleanly replaces Turn 1 (no accumulation)
    state = replace_on_new_turn_reducer(state, [p2])
    assert len(state) == 1
    assert state[0].go_number == "GO-2"

    # Reset on empty retrieval
    state = replace_on_new_turn_reducer(state, [])
    assert len(state) == 0


# ===========================================================================
# 6. Last-Write-Wins Reducer Tests
# ===========================================================================

def test_last_write_wins_fields():
    """Test last-write-wins replacement semantics on scalar and object fields."""
    # query_text
    q = last_write_wins_reducer("Query 1", "Query 2")
    assert q == "Query 2"

    # query_language
    lang = last_write_wins_reducer("hi", "hinglish")
    assert lang == "hinglish"

    # query_filters
    f1 = QueryFilters(department="Forest")
    f2 = QueryFilters(department="Revenue", year_range=[2018, 2020])
    filters = last_write_wins_reducer(f1, f2)
    assert filters.department == "Revenue"
    assert filters.year_range == [2018, 2020]

    # confidence_score
    score = last_write_wins_reducer(0.5, 0.92)
    assert score == 0.92

    # supersession_status
    status = last_write_wins_reducer("UNKNOWN", "CURRENT_ACTIVE")
    assert status == "CURRENT_ACTIVE"

    # human_verification
    app1 = ApprovalState(action="approve", resolved_go_number="GO-100")
    app_state = last_write_wins_reducer(None, app1)
    assert app_state.action == "approve"

    # graceful_refusal
    refusal = last_write_wins_reducer(False, True)
    assert refusal is True


# ===========================================================================
# 7. Strict Pydantic V2 Schema Validation Tests
# ===========================================================================

def test_strict_mode_forbids_extra_fields():
    """Verify strict mode forbids unmodeled extra attributes."""
    with pytest.raises(ValidationError):
        OfficerContext(department="Forest", access_scope=["Forest"], extra_unauthorized_key="bypass")


def test_confidence_score_range_validation():
    """Verify confidence_score enforces [0.0, 1.0] bounds."""
    # Valid
    assessment = ConfidenceSupersessionAssessment(
        confidence_score=0.75,
        supersession_status="CURRENT_ACTIVE",
        personal_data_flag=False,
        requires_deep_reasoning=False,
    )
    assert assessment.confidence_score == 0.75

    # Out of bounds (> 1.0)
    with pytest.raises(ValidationError):
        ConfidenceSupersessionAssessment(
            confidence_score=1.5,
            supersession_status="CURRENT_ACTIVE",
            personal_data_flag=False,
            requires_deep_reasoning=False,
        )

    # Out of bounds (< 0.0)
    with pytest.raises(ValidationError):
        ConfidenceSupersessionAssessment(
            confidence_score=-0.1,
            supersession_status="CURRENT_ACTIVE",
            personal_data_flag=False,
            requires_deep_reasoning=False,
        )


def test_grounded_answer_requires_non_empty_citations():
    """Verify GroundedAnswer strictly enforces min_length=1 on citations."""
    cit = Citation(
        go_number="GO-1", issuing_department="Dept", date="2020-01-01", page_number=1, exact_text_excerpt="Excerpt"
    )
    # Valid with citation
    ans = GroundedAnswer(answer_markdown="Valid answer text.", citations=[cit])
    assert len(ans.citations) == 1

    # Invalid with empty citations
    with pytest.raises(ValidationError):
        GroundedAnswer(answer_markdown="Ungrounded answer.", citations=[])


def test_scope_screen_decision_enums():
    """Verify ScopeScreenDecision category allowlist."""
    valid_dec = ScopeScreenDecision(
        in_scope=False, category="financial_disbursement", reason="Treasury request"
    )
    assert valid_dec.category == "financial_disbursement"

    with pytest.raises(ValidationError):
        ScopeScreenDecision(in_scope=False, category="unauthorized_category", reason="Invalid")
