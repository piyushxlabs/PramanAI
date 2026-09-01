"""Indic Legacy (KrutiDev & Chanakya) to Unicode Devanagari Converter for Uttarakhand Government Orders.

Converts legacy Hindi Kruti Dev 010/011 and Walkman Chanakya ASCII-encoded font text
extracted from official Uttarakhand GO PDFs into clean, standardized Devanagari Unicode.
"""

import re
from typing import Final

# Common Chanakya / Walkman Indic phrase replacements
CHANAKYA_PHRASE_MAP: Final[list[tuple[str, str]]] = [
    (r"T4T1-4ff\s*4\.flT-I", "उत्तराखण्ड शासन"),
    (r"31-800-\)1=1-R-", "शासनादेश"),
    (r"49'\s*'cr44\s*4\s*cNul\s*3ITTITT-2", "वन विभाग अनुभाग-2"),
    (r"49'\s*'cr44\s*4\s*cNul", "वन विभाग"),
    (r"3ITTITT-2", "अनुभाग-2"),
    (r"3ITTITT", "अनुभाग"),
    (r"cNul", "विभाग"),
    (r"\*ft-4W", "देहरादून"),
    (r"3FRTF", "अगस्त"),
    (r"itdrzi\s*44", "वित्तीय वर्ष"),
    (r"itdrzi\s*citi", "वित्तीय वर्ष"),
    (r"itdrzi\.", "वित्तीय"),
    (r"itdrzi", "वित्तीय"),
    (r"311-4T9", "अनुदान"),
    (r"34-1-4T-9-", "अनुदान"),
    (r"378-4T9-", "अनुदान"),
    (r"TrtY4zr", "श्री राज्यपाल"),
    (r"act-cirw", "स्वीकृति"),
    (r"raisr", "प्रदान"),
    (r"anzrwrzi", "अनुमोदन"),
    (r"anzr—wrzi", "अनुमोदन"),
    (r"c414,", "भवन"),
    (r"Er-l", "मार्ग"),
    (r"070 - #9-Rfferrli79-", "070 - संचार"),
    (r"2406 -ar", "2406 - वानिकी"),
    (r"MIT 471-T", "लेखा शीर्षक"),
    (r"\*1:0-27", "सं०-27"),
    (r"TZTT", "द्वारा"),
    (r"\b11\b", "में"),
    (r"citi", "वर्ष"),
    (r"1i,1", "दिनांक"),
]

# Multi-character substitutions applied first for KrutiDev
KRUTIDEV_MULTI_MAP: Final[list[tuple[str, str]]] = [
    ("mR", "उत्"),
    ("m", "उ"),
    ("vkS", "औ"),
    ("vks", "ओ"),
    ("vk", "आ"),
    ("v", "अ"),
    ("bZ", "ई"),
    ("b", "इ"),
    ("Å", "ऊ"),
    ("ए", "ए"),
    ("ks", "ो"),
    ("kS", "ौ"),
    ("k", "ा"),
    ("'", "श"),
    ('"', "ष"),
    ("['", "ख"),
    ("[k", "ख"),
    ("[", "ख"),
    ("Fk", "थ"),
    ("Hk", "भ"),
    ("Pk", "छ"),
    (".k", "ण"),
    ("=", "त्र"),
    ("{'", "श्र"),
    ("K", "ज्ञ"),
    ("Ø", "क्र"),
    ("Í", "द्र"),
    ("Î", "द्द्र"),
    ("Ï", "ष्ट"),
    ("Ñ", "कृ"),
    ("Ô", "ष्ठ"),
    ("Õ", "ट्ट"),
    ("Ö", "ट्ठ"),
    ("Ù", "द्द"),
    ("Ú", "द्ध"),
    ("Û", "द्य"),
    ("Ü", "द्व"),
    ("Ý", "फ्र"),
]

# Single character substitutions for KrutiDev
KRUTIDEV_SINGLE_MAP: Final[dict[str, str]] = {
    "d": "क",
    "D": "क्",
    "x": "ग",
    "X": "ग्",
    "?": "घ",
    "p": "च",
    "P": "च्",
    "N": "छ",
    "t": "ज",
    "T": "ज्",
    "÷": "झ",
    "V": "ट",
    "B": "ठ",
    "M": "ड",
    "<": "ढ",
    ".": "ण्",
    "r": "त",
    "R": "त्",
    "F": "थ",
    "n": "द",
    "/": "ध",
    "u": "न",
    "U": "न्",
    "i": "प",
    "I": "प्",
    "Q": "फ",
    "c": "ब",
    "C": "ब्",
    "H": "भ",
    "e": "म",
    "E": "म्",
    ";": "य",
    "Y": "य्",
    "j": "र",
    "y": "ल",
    "o": "व",
    "O": "व्",
    "l": "स",
    "L": "स्",
    "g": "ह",
    "h": "ी",
    "q": "ु",
    "w": "ू",
    "s": "े",
    "S": "ै",
    "a": "ं",
    "A": "ँ",
    ":": "ः",
    "W": "ॅ",
    "~": "्",
    "`": "ृ",
    "+": "्",
    "1": "१",
    "2": "२",
    "3": "३",
    "4": "४",
    "5": "५",
    "6": "६",
    "7": "७",
    "8": "८",
    "9": "९",
    "0": "०",
}

# Common indicator words frequently appearing in Uttarakhand GOs
_INDIC_LEGACY_SAMPLE_WORDS = [
    "mRrjk[k.M",  # उत्तराखण्ड (KrutiDev)
    "'kklu",       # शासन (KrutiDev)
    "la[;k",       # संख्या (KrutiDev)
    "ou foHkkx",   # वन विभाग (KrutiDev)
    "foHkkx",      # विभाग (KrutiDev)
    "fnukad",      # दिनांक (KrutiDev)
    "nsgjknwu",    # देहरादून (KrutiDev)
    "vkns'k",      # आदेश (KrutiDev)
    "vf/klwpuk",   # अधिसूचना (KrutiDev)
    "dk;kZy;",     # कार्यालय (KrutiDev)
    "jktkKk",      # शासनादेश (KrutiDev)
    "mRrj",        # उत्तर (KrutiDev)
    "31-800-)1=1-R-", # शासनादेश (Chanakya)
    "T4T1-4ff",    # उत्तराखण्ड (Chanakya)
    "4.flT-I",     # शासन (Chanakya)
    "itdrzi",      # वित्तीय (Chanakya)
    "311-4T9",     # अनुदान (Chanakya)
    "TrtY4zr",     # राज्यपाल (Chanakya)
    "act-cirw",    # स्वीकृति (Chanakya)
]


def is_likely_indic_legacy(text: str) -> bool:
    """Returns True if the text contains signatures of KrutiDev or Chanakya font encoding."""
    if not text:
        return False
    devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    if devanagari_chars > len(text) * 0.3:
        return False

    for sample in _INDIC_LEGACY_SAMPLE_WORDS:
        if sample in text:
            return True

    if "foHk" in text or "mRr" in text or "'kkl" in text or "la[" in text or "fnuk" in text:
        return True
    if "itdrzi" in text or "T4T1" in text or "311-4T9" in text or "TrtY" in text:
        return True

    return False


def is_likely_krutidev(text: str) -> bool:
    """Backwards-compatible alias for is_likely_indic_legacy."""
    return is_likely_indic_legacy(text)


def chanakya_to_unicode(text: str) -> str:
    """Converts a Chanakya/Walkman encoded string into clean Devanagari Unicode."""
    if not text:
        return ""
    # Strip typist file paths (e.g., D:\Rakesh Mahar\...)
    modified = re.sub(r"[A-Za-z]:\\[^\n]+", "", text)
    for pattern, repl in CHANAKYA_PHRASE_MAP:
        modified = re.sub(pattern, repl, modified)
    return modified


def krutidev_to_unicode(text: str) -> str:
    """Converts a KrutiDev 010/011 encoded string into clean Devanagari Unicode."""
    if not text:
        return ""

    modified_text = text

    # Handle Reph character Z/z
    modified_text = re.sub(r"([a-zA-Z\[\<\>\/\?\.\;]+)Z", r"र्\1", modified_text)
    modified_text = re.sub(r"([a-zA-Z\[\<\>\/\?\.\;]+)z", r"र्\1", modified_text)

    # Multi-character mappings
    for kd, uni in KRUTIDEV_MULTI_MAP:
        modified_text = modified_text.replace(kd, uni)

    # Single character mappings
    converted_chars = []
    for char in modified_text:
        converted_chars.append(KRUTIDEV_SINGLE_MAP.get(char, char))
    modified_text = "".join(converted_chars)

    # Chhoti 'i' matra repositioning
    modified_text = re.sub(r"f([क-ह](?:्[क-ह])?)", r"\1ि", modified_text)
    modified_text = re.sub(r"ि([क-ह](?:्[क-ह])?)", r"\1ि", modified_text)

    # Clean double matras
    modified_text = modified_text.replace("िि", "ि")
    modified_text = modified_text.replace("ाा", "ा")
    modified_text = modified_text.replace("ीी", "ी")

    return modified_text


def convert_if_krutidev(text: str) -> str:
    """Detects legacy Indic (KrutiDev/Chanakya) text and automatically converts to Unicode Devanagari."""
    if not text:
        return ""
    # First, strip typist paths
    text = re.sub(r"[A-Za-z]:\\[^\n]+", "", text)
    
    # Apply Chanakya substitutions if present
    text = chanakya_to_unicode(text)

    # Apply KrutiDev conversion if KrutiDev signatures exist
    if is_likely_krutidev(text):
        text = krutidev_to_unicode(text)

    return text
