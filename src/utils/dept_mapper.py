"""Bilingual taxonomy mapping module for Uttarakhand Government departments.

Expands English and Hindi department names into bilingual keywords for robust SQL filtering.
"""

from typing import List

DEPARTMENT_MAP = {
    "forest": ["वन", "पर्यावरण", "forest", "wildlife", "van"],
    "general_admin": ["सामान्य प्रशासन", "gad", "general administration", "कार्मिक", "general"],
    "personnel": ["कार्मिक", "personnel", "नियुक्ति", "karmik"],
    "revenue": ["राजस्व", "revenue", "तहसील"],
    "education": ["शिक्षा", "इण्टर कालेज", "education", "विद्यालय", "shiksha"],
    "finance": ["वित्त", "finance", "कोषागार", "लेखा", "pension", "वेतन"],
}


def get_dept_keywords(dept_input: str) -> List[str]:
    """Expands English or Hindi department names into bilingual search keywords."""
    if not dept_input:
        return []
    clean_input = dept_input.strip().lower()
    for key, variations in DEPARTMENT_MAP.items():
        if clean_input in key or any(clean_input == v.lower() or v.lower() in clean_input for v in variations):
            return variations
    return [dept_input]
