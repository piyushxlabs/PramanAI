"""Input sanitization and parameter governance for ShasanAI tools.

Implements defense-in-depth sanitization per AGENT_LOGIC_SPEC.md Section 8:
- Rejects SQL-metacharacter-like (;, --, /*, xp_, EXEC) and path-traversal (../, ..\\) sequences.
- Enforces 500-character maximum query length constraint.
- Validates department and policy category filters against known allowlists.
- Validates year range constraints (1950 <= start_year <= end_year <= current_year).
- Injects officer_context.access_scope server-side (never model-supplied or widenable).
"""

from datetime import datetime
import re
from typing import Optional
from src.state.reducers import ScopeViolationError, StateValidationError
from src.state.schema import OfficerContext

import logging

logger = logging.getLogger("shasanai.sanitization")

# SQL metacharacter and injection patterns
SQL_INJECTION_PATTERN = re.compile(
    r"(;|--|/\*|\*/|\bxp_\b|\bexec\b|\bunion\b|\bselect\b|\bdrop\b|\binsert\b|\bupdate\b|\bdelete\b)",
    re.IGNORECASE,
)

# Path traversal patterns
PATH_TRAVERSAL_PATTERN = re.compile(
    r"(\.\./|\.\.\\|/etc/|c:\\|/var/|/root/)",
    re.IGNORECASE,
)

# Known Uttarakhand Administrative Departments allowlist
KNOWN_DEPARTMENTS: set[str] = {
    "Forest",
    "Revenue",
    "Finance",
    "Education",
    "Health",
    "Home",
    "Personnel",
    "Public Works",
    "Rural Development",
    "Urban Development",
    "ITDA",
    "General Administration",
    "Agriculture",
    "Transport",
    "Energy",
    "Tourism",
    "Social Welfare",
    "Irrigation",
    "Food & Civil Supplies",
    "General",
}

# Known Administrative Policy Categories allowlist
KNOWN_POLICY_CATEGORIES: set[str] = {
    "Transfer Policy",
    "Pay Scale",
    "Regularization",
    "Leave Rules",
    "Pension & Gratuity",
    "Procurement",
    "Service Rules",
    "Reservation",
    "Disciplinary Action",
    "Promotion",
    "Recruitment",
    "General Circular",
    "Financial Sanction",
    "Budget Allocation",
    "Budget",
    "Financial Allocation",
    "Grant",
    "Pension",
}

# Policy category synonym normalization map
POLICY_CATEGORY_SYNONYMS: dict[str, str] = {
    "budget": "Budget Allocation",
    "budget allocation": "Budget Allocation",
    "financial allocation": "Budget Allocation",
    "grant": "Budget Allocation",
    "financial sanction": "Financial Sanction",
    "sanction": "Financial Sanction",
    "transfer": "Transfer Policy",
    "posting": "Transfer Policy",
    "transfer policy": "Transfer Policy",
    "regularization": "Regularization",
    "samayojan": "Regularization",
    "pension": "Pension & Gratuity",
    "gratuity": "Pension & Gratuity",
    "pension & gratuity": "Pension & Gratuity",
    "retirement": "Pension & Gratuity",
    "leave": "Leave Rules",
    "leave rules": "Leave Rules",
    "holiday": "Leave Rules",
    "recruitment": "Recruitment",
    "vacancy": "Recruitment",
}


def sanitize_query_text(query: str, max_length: int = 500) -> str:
    """Validates and sanitizes free-text query string.
    
    Raises:
        StateValidationError: If query exceeds max length, is empty, or contains injection patterns.
    """
    if not query or not query.strip():
        raise StateValidationError("Query text cannot be empty or whitespace only")

    trimmed = query.strip()
    if len(trimmed) > max_length:
        raise StateValidationError(
            f"Query text length ({len(trimmed)}) exceeds maximum allowed ({max_length} characters)"
        )

    if SQL_INJECTION_PATTERN.search(trimmed):
        raise StateValidationError("Query text contains forbidden SQL metacharacters or command sequences")

    if PATH_TRAVERSAL_PATTERN.search(trimmed):
        raise StateValidationError("Query text contains forbidden path traversal sequences")

    return trimmed


def sanitize_department(department: Optional[str]) -> Optional[str]:
    """Validates department filter against allowlist with resilient fallback."""
    if department is None or department.strip() == "":
        return None

    clean = department.strip()
    # Check injection patterns first (security gate)
    if SQL_INJECTION_PATTERN.search(clean) or PATH_TRAVERSAL_PATTERN.search(clean):
        raise StateValidationError(f"Invalid department filter syntax: '{clean}'")

    # Match against allowlist (case-insensitive search, canonical return)
    for known in KNOWN_DEPARTMENTS:
        if known.lower() == clean.lower():
            return known

    # Resilient fallback: log warning and ignore unrecognized benign filter rather than crashing
    logger.warning(
        "Department '%s' not in recognized allowlist; falling back to None for broad search.", clean
    )
    return None


def sanitize_policy_category(category: Optional[str]) -> Optional[str]:
    """Validates policy category filter against allowlist and synonym map with resilient fallback."""
    if category is None or category.strip() == "":
        return None

    clean = category.strip()
    # Check injection patterns first (security gate)
    if SQL_INJECTION_PATTERN.search(clean) or PATH_TRAVERSAL_PATTERN.search(clean):
        raise StateValidationError(f"Invalid policy category filter syntax: '{clean}'")

    # 1. Exact / case-insensitive match in allowlist
    for known in KNOWN_POLICY_CATEGORIES:
        if known.lower() == clean.lower():
            return known

    # 2. Synonym map resolution
    clean_lower = clean.lower()
    if clean_lower in POLICY_CATEGORY_SYNONYMS:
        return POLICY_CATEGORY_SYNONYMS[clean_lower]

    for syn_key, canonical in POLICY_CATEGORY_SYNONYMS.items():
        if syn_key in clean_lower or clean_lower in syn_key:
            return canonical

    # 3. Resilient fallback: log warning and return None rather than raising StateValidationError
    logger.warning(
        "Policy category '%s' not in recognized allowlist; falling back to None for broad search.", clean
    )
    return None


def sanitize_year_range(year_range: Optional[list[int]]) -> Optional[list[int]]:
    """Validates year range [start_year, end_year] bounds."""
    if year_range is None:
        return None

    if len(year_range) != 2:
        raise StateValidationError(f"Year range filter must contain exactly [start_year, end_year], got: {year_range}")

    start_year, end_year = year_range[0], year_range[1]
    current_year = datetime.now().year + 1  # Allow current/next legislative year

    if start_year < 1950 or end_year < 1950:
        raise StateValidationError(f"Year range ({start_year}, {end_year}) cannot precede 1950")

    if start_year > current_year or end_year > current_year:
        raise StateValidationError(f"Year range ({start_year}, {end_year}) cannot exceed current year ({current_year})")

    if start_year > end_year:
        raise StateValidationError(f"Invalid year range: start_year ({start_year}) > end_year ({end_year})")

    return [start_year, end_year]


def inject_server_access_scope(officer_context: OfficerContext, requested_dept: Optional[str] = None) -> list[str]:
    """Server-side access scope resolution.
    
    Verifies that any requested department filter is within the officer's authorized access scope.
    """
    authorized_scopes = officer_context.access_scope or [officer_context.department]

    if requested_dept:
        if requested_dept not in authorized_scopes and "ALL" not in authorized_scopes:
            raise ScopeViolationError(
                f"Officer from '{officer_context.department}' is not authorized to query department '{requested_dept}'. Authorized: {authorized_scopes}"
            )
        return [requested_dept]

    return authorized_scopes
