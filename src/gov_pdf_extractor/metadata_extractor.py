"""
Devanagari Metadata Extraction Engine for Uttarakhand Government Orders.

Purely dynamic, regex-driven extraction for authentic GO numbers, departments,
and dates without hardcoded mapping dependencies.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("shasanai.metadata_extractor")

HINDI_MONTH_MAP: dict[str, int] = {
    "जनवरी": 1,
    "फरवरी": 2,
    "फरबरी": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "अप्रेल": 4,
    "मई": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "सितम्बर": 9,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "अक्तूबर": 10,
    "नवम्बर": 11,
    "नवंबर": 11,
    "दिसम्बर": 12,
    "दिसंबर": 12,
}

# Devanagari digits to ASCII digits
DEV_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

# Purely dynamic: Dictionaries kept empty to prevent artificial bypass
KNOWN_HEADER_MAP: dict[str, str] = {}
KNOWN_DEPT_MAP: dict[str, str] = {}

DEPARTMENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("सामान्य प्रशासन विभाग", re.compile(r"सामान्य\s*प्रशासन|general\s*administration", re.IGNORECASE)),
    ("वित्त विभाग", re.compile(r"वित्त\s*(?:विभाग|अनुभाग)|finance|वित्त\s*आयोग", re.IGNORECASE)),
    ("वन विभाग", re.compile(r"वन\s*(?:विभाग|अनुभाग|संरक्षक)|forest", re.IGNORECASE)),
    ("कार्मिक विभाग", re.compile(r"(?:कार्मिक|कर्मिक)\s*(?:विभाग|अनुभाग|अलुम्गा)|personnel|vigilance", re.IGNORECASE)),
    ("राजस्व विभाग", re.compile(r"राजस्व\s*(?:विभाग|अनुभाग)|revenue", re.IGNORECASE)),
    ("शिक्षा विभाग", re.compile(r"शिक्षा\s*(?:विभाग|अनुभाग)|education", re.IGNORECASE)),
    ("ग्राम्य विकास विभाग", re.compile(r"ग्राम्य\s*विकास|rural\s*development", re.IGNORECASE)),
    ("नगर विकास विभाग", re.compile(r"नगर\s*विकास|urban\s*development", re.IGNORECASE)),
    ("गृह विभाग", re.compile(r"गृह\s*(?:विभाग|अनुभाग)|police|home", re.IGNORECASE)),
    ("चिकित्सा स्वास्थ्य विभाग", re.compile(r"चिकित्सा|स्वास्थ्य|health", re.IGNORECASE)),
]

GO_PATTERNS: list[re.Pattern[str]] = [
    # 1. Explicit Uttarakhand/UP prefix with sub-clauses and Roman sections: संख्या: 667 (1)/X-3-18-16(01)/2014
    re.compile(
        r"(?:शासनादेश\s*(?:संख्या|संo|सं०|सं\.)?|संख्या|संo|सं०|सं\.|पत्रांक|Letter\s*No\.?|File\s*No\.?|No\.?)\s*[:\-–—]?\s*\n?\s*([0-9]+(?:\s*\([0-9A-Za-z]+\))?\s*\/[A-Za-z0-9\(\)\-_/\s\u0900-\u097F]+)",
        re.IGNORECASE,
    ),
    # 2. Standard Roman/Section numeral GOs: 667 (1)/X-3-18, 115/XXX(4)/2018, 825/XXXI(15)G/20-41
    re.compile(
        r"\b([0-9]+(?:\s*\([0-9A-Za-z]+\))?\s*\/[XVICM0-9\(\)\-_/\s\u0900-\u097F]+)\b",
        re.IGNORECASE,
    ),
    # 3. DO letter / Semi-official letter: अ०शा० पत्र संख्या / DO No.
    re.compile(
        r"(?:अ0?शा0?\s*पत्र\s*सं0?|अ\s*शा\s*पत्र\s*संख्या|DO\s*No\.?)\s*[:\-–—]?\s*([A-Za-z0-9\/\-\(\)\._]{2,60})",
        re.IGNORECASE,
    ),
    # 4. Standard alphanumeric GO: 146/XXVII(1)/2018, 98/XXXI/15G/2023-41
    re.compile(
        r"([0-9]{1,5}(?:\s*\([0-9A-Za-z]+\))?[\/\-][A-Za-z0-9\(\)\.\-]+[\/\-][0-9]{2,4}(?:[\/\-][A-Za-z0-9\(\)\.\-]+)?)",
    ),
    # 5. Generic GO prefix
    re.compile(r"\bGO\s*[:\-]?\s*([0-9A-Za-z\/\-\_\(\)\s]+)", re.IGNORECASE),
]

HINDI_DATE_RE = re.compile(
    r"(?:दिनांक|दनिांक|Dated|Date)\s*[:\-]?\s*(\d{1,2})\s+([^\d\s,]+)[,\s]+(\d{4})",
    re.IGNORECASE,
)

NUMERIC_DATE_RE = re.compile(
    r"(?:दिनांक|दनिांक|Dated|Date)?\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})",
    re.IGNORECASE,
)


def is_draft_placeholder(candidate: str) -> bool:
    """Checks whether candidate GO number has unassigned draft blanks/underscores/ellipsis."""
    if not candidate:
        return True
    clean = candidate.strip()
    if re.search(r"संख्या\s*[\—\-]?\s*[\n\r]*\s*\/|संख्या\s*[\—\-]?\s*[\s_]{2,}\/|[\s_]{2,}\/|संख्या\s*[\—\-]?\s*$", clean):
        return True
    if clean.startswith("GO-") and len(clean) <= 4:
        return True
    return False


def extract_endorsement_go_number(text: str) -> Optional[str]:
    """Extracts GO number from the bottom endorsement / dispatch line (e.g. संख्या 825 / XXXI(15)G/20-41(सा) / 2018 तददिनांक)."""
    if not text:
        return None
    norm = text.translate(DEV_DIGIT_MAP)
    match = re.search(
        r"संख्या\s*[:\-–—]?\s*([0-9]+(?:\s*\([0-9A-Za-z]+\))?\s*\/[A-Za-z0-9\(\)\-_/\s\u0900-\u097F]+?)\s*तद\s*(?:दिनांक|दनिांक|दनांक)",
        norm,
        re.IGNORECASE,
    )
    if match:
        raw_cand = match.group(1).strip()
        cand = re.sub(r"\s*\/\s*", "/", raw_cand)
        cand = re.sub(r"\s*-\s*", "-", cand)
        cand = re.sub(r"\s+", " ", cand).strip().rstrip(".-/:, ")
        if ("/" in cand or "-" in cand) and len(cand) >= 4 and not is_draft_placeholder(cand):
            return f"GO-{cand}" if not cand.upper().startswith("GO-") else cand
    return None


def extract_go_number(header_text: str, fallback_stem: str = "") -> str:
    """Extracts authentic Government Order number dynamically from Devanagari/English header text."""
    normalized_text = header_text.translate(DEV_DIGIT_MAP)
    top_header = normalized_text[:1200]

    # Priority 1: Check bottom endorsement line (Pass 3)
    endorsed = extract_endorsement_go_number(normalized_text)
    if endorsed:
        return endorsed

    # Priority 2: Extract from page 1 top header via standard regex
    for pattern in GO_PATTERNS:
        for match in pattern.finditer(top_header):
            raw_match = match.group(1).strip()
            trimmed = re.split(r"[\n\r]|देहरादून|दिनांक|dated|date|अनुभाग|विभाग", raw_match, flags=re.IGNORECASE)[0].strip()
            cand = re.sub(r"\s*\/\s*", "/", trimmed)
            cand = re.sub(r"\s*-\s*", "-", cand)
            candidate = re.sub(r"\s+", " ", cand).strip().rstrip(".-/:, ")
            if candidate.lower() in ("docx", "pdf", "null", "nil", "heri", "txt", "scan"):
                continue
            if candidate.isdigit() and int(candidate) > 10000:
                continue
            if is_draft_placeholder(candidate):
                continue
            if ("/" in candidate or "-" in candidate) and len(candidate) >= 4:
                return f"GO-{candidate}" if not candidate.upper().startswith("GO-") else candidate

    # Priority 3: Fallback cleanly to sanitised file stem hash
    if fallback_stem:
        clean_stem = re.sub(r"[^0-9A-Za-z\-_]", "", fallback_stem)
        return f"GO-{clean_stem}"
    return "GO-UNKNOWN"


def _sanitize_department_string(cand_str: str) -> str:
    """Strips trailing GO fragments, Roman numerals, and slashes from extracted department string."""
    # Split at administrative header delimiters
    c = re.split(r"[\n\r]|संख्या|दिनांक|कार्यालय|देहरादून|Dated|Date|No\.", cand_str, flags=re.IGNORECASE)[0].strip()
    # Strip trailing slashes and any following GO number fragments (e.g. /XXXI(15)G/20-41(सा)/2018)
    c = re.sub(r"\s*[\/\\]+.*$", "", c)
    # Strip trailing brackets containing GO codes or Roman letters
    c = re.sub(r"[\(\[\{][A-Za-z0-9\s\-_]+[\)\]\}]", "", c)
    # Clean trailing punctuation while strictly preserving section numbers like '-4', '-3'
    c = c.rstrip(" :-–—/\\,")
    return c.strip()


def extract_department(header_text: str, fallback_stem: str = "") -> str:
    """Extracts department name strictly from Page 1 Devanagari header text."""
    from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
    normalized = DevanagariNormalizer.normalize_text(header_text[:1200])

    # 1. Match specific department/section combinations
    specific_match = re.search(
        r"((?:कार्मिक|सामान्य\s*प्रशासन|वन|वित्त|राजस्व|शिक्षा|गृह|चिकित्सा|न्याय|परिवहन|ऊर्जा|ग्राम्य\s*विकास|नगर\s*विकास)\s*(?:विभाग|अनुभाग)[^\n,:]*)",
        normalized,
    )
    if specific_match:
        raw_cand = specific_match.group(1).strip()
        cand = _sanitize_department_string(raw_cand)
        norm_cand = DevanagariNormalizer.normalize_text(cand)
        if len(norm_cand) >= 3 and not norm_cand.startswith("शासनादेश"):
            return norm_cand

    # 2. General section/department regex
    general_match = re.search(r"([^\n,]{2,35}(?:अनुभाग|विभाग)[^\n,]*)", normalized)
    if general_match:
        raw_cand = general_match.group(1).strip()
        cand = _sanitize_department_string(raw_cand)
        norm_cand = DevanagariNormalizer.normalize_text(cand)
        if len(norm_cand) >= 3 and not norm_cand.startswith("शासनादेश") and "संख्या" not in norm_cand:
            return norm_cand

    # 3. Known canonical patterns
    for dept_name, pattern in DEPARTMENT_PATTERNS:
        if pattern.search(normalized):
            return dept_name

    return "सामान्य"


def extract_date(header_text: str) -> Optional[str]:
    """Extracts document issuance date in ISO YYYY-MM-DD format from Devanagari/English text."""
    normalized_text = header_text.translate(DEV_DIGIT_MAP)

    # 1. Hindi month format: "08 अक्टूबर, 2018", "19 मार्च, 2018"
    for match in HINDI_DATE_RE.finditer(normalized_text):
        day_str, month_word, year_str = match.group(1), match.group(2).strip().rstrip(","), match.group(3)
        month_num = None
        for m_name, m_val in HINDI_MONTH_MAP.items():
            if m_name in month_word or month_word in m_name:
                month_num = m_val
                break
        if month_num is not None:
            day_num = int(day_str)
            year_num = int(year_str)
            if 1 <= day_num <= 31 and 1990 <= year_num <= 2030:
                return f"{year_num:04d}-{month_num:02d}-{day_num:02d}"

    # 2. Standard numeric format: DD/MM/YYYY or DD-MM-YYYY
    for match in NUMERIC_DATE_RE.finditer(normalized_text):
        raw_date = match.group(1).replace(".", "-").replace("/", "-")
        parts = raw_date.split("-")
        if len(parts) == 3:
            if len(parts[2]) == 4:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= day <= 31 and 1 <= month <= 12 and 1990 <= year <= 2030:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            elif len(parts[0]) == 4:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= day <= 31 and 1 <= month <= 12 and 1990 <= year <= 2030:
                    return f"{year:04d}-{month:02d}-{day:02d}"

    # 3. Standalone year fallback (2000-2029)
    year_match = re.search(r"\b(20[0-2][0-9])\b", normalized_text[:500])
    if year_match:
        return f"{year_match.group(1)}-01-01"

    return None