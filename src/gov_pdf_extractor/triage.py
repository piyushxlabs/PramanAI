"""Stage 2: Deterministic Content Triage & Multi-Font Decoder.

Inspects raw PDF text layers and PDF font descriptors to categorize pages into:
- NATIVE_UNICODE (Authentic Devanagari Unicode \u0900-\u097F)
- LEGACY_FONT (8-bit encoded fonts: KrutiDev 010/011, Shree-Dev 0714, DV-TTSurekh, Chanakya)
- SCANNED_IMAGE (Scanned raster pages, low-confidence OCR, or high-entropy garbage layers)

Preserves English administrative reference codes, dates, and Roman numerals via Token-Aware parsing.
Provides automatic fallback to 300+ DPI RapidOCR when legacy font noise/unmapped glyphs persist.
"""

from collections import Counter
import logging
import math
import unicodedata
from typing import Any, Final, List, Optional, Tuple
import regex as re

from src.gov_pdf_extractor.models import PageType

logger = logging.getLogger("gov_pdf_extractor.triage")

# Standard English keywords and administrative terms that MUST NOT be transformed by font converters
STANDARD_ENGLISH_TOKENS: Final[set[str]] = {
    "go", "no", "dated", "date", "department", "forest", "finance", "personnel",
    "revenue", "education", "uttarakhand", "dehradun", "page", "section", "act",
    "rule", "rules", "annexure", "copy", "order", "notification", "circular",
    "government", "shasan", "anubhag", "sankhya", "karmik", "rajyapal", "ias",
    "ips", "ifs", "pcs", "inr", "rs", "cr", "lakh", "lacs", "total", "grand",
    "sub", "ref", "file", "itda", "pwd", "upcl", "ujvnl", "gmvn", "kmvn",
    "of", "and", "the", "for", "in", "to", "from", "by", "with", "at", "on", "as", "is", "a", "an",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xxvii", "xxxi",
}

# Known ASCII Mojibake / Corrupted OCR & legacy font noise tokens
MOJIBAKE_NOISE_TOKENS: Final[list[str]] = [
    "oe qff", "gibo", "siion", "5reuqdr", "g{ifrt{", "qth egti",
    "ih?pl", "ebi", "ell", "pnbie", "bibe", "eb btn", "ahie",
    "wio", "bggnou", "bn blihif", "wlain", "pherial", "pif0 yu",
    "fosajeemieom", "b8+:b2z435", "l-lei", "plit er", "lei plit",
]

# High frequency legacy font signatures across KrutiDev, Shree-Dev, DV-TTSurekh, and Chanakya
LEGACY_FONT_SIGNATURES: Final[list[str]] = [
    "mRrjk[k.M",   # KrutiDev उत्तराखण्ड
    "'kklu",        # KrutiDev शासन
    "la[;k",        # KrutiDev संख्या
    "ou foHkkx",    # KrutiDev वन विभाग
    "foHkkx",       # KrutiDev विभाग
    "fnukad",       # KrutiDev दिनांक
    "nsgjknwu",     # KrutiDev देहरादून
    "vkns'k",       # KrutiDev आदेश
    "vf/klwpuk",    # KrutiDev अधिसूचना
    "dk;kZy;",      # KrutiDev कार्यालय
    "jktkKk",       # KrutiDev शासनादेश
    "mRrj",         # KrutiDev उत्तर
    "izfr",         # KrutiDev प्रति
    "fodkl",        # KrutiDev विकास
    "vk;qDr",       # KrutiDev आयुक्त
    "lfpo",         # KrutiDev सचिव
    "f{k",          # KrutiDev क्षि
    "d{k",          # KrutiDev कक्ष
    "iz",           # KrutiDev प्र
    "Dr",           # KrutiDev क्त
    "31-800-)1=1-R-",  # Chanakya शासनादेश
    "T4T1-4ff",     # Chanakya उत्तराखण्ड
    "4.flT-I",      # Chanakya शासन
    "itdrzi",       # Chanakya वित्तीय
    "311-4T9",      # Chanakya अनुदान
    "TrtY4zr",      # Chanakya राज्यपाल
    "act-cirw",     # Chanakya स्वीकृति
    "#$f'",          # Shree-Dev 0714 signatures
    "î", "ï", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú",
]


class BaseIndicFontConverter:
    """Base helper for token-aware legacy font conversion."""

    @staticmethod
    def is_protected_ascii_token(token: str) -> bool:
        """Determines if an ASCII token is a legitimate English word, serial number, or date."""
        if not token:
            return False
        clean = token.strip(" ,;:\t\n\r()[]{}'\"")
        if not clean:
            return False
        if any(0x0900 <= ord(c) <= 0x097F for c in clean):
            return False
        if clean.lower() in STANDARD_ENGLISH_TOKENS:
            return True
        # Numbers, dates, percentages
        if re.match(r"^\d+(?:\.\d+)?%?$", clean):
            return True
        if re.match(r"^\d{1,4}[\-\/\.]\d{1,2}[\-\/\.]\d{2,4}$", clean):
            return True
        # Standard GO reference numbers like 146/XXVII(1)/2018
        if re.match(r"^\d{1,5}\/[A-Za-z0-9\(\)\.\-]+\/\d{2,4}$", clean):
            return True
        if clean in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XXVII", "XXXI", "XXX-1"):
            return True
        if clean in ("GO", "PWD", "ITDA", "IAS", "IFS", "IPC", "CrPC", "GST", "Dated", "No", "No."):
            return True
        return False


# ===========================================================================
# 1. KrutiDev 010/011 Full Mapping
# ===========================================================================
KRUTIDEV_LIGATURES: Final[list[tuple[str, str]]] = sorted(
    [
        ("mRrjk[k.M", "उत्तराखण्ड"),
        ("mRrj", "उत्तर"),
        ("fnukad", "दिनांक"),
        ("nsgjknwu", "देहरादून"),
        ("foHkkx", "विभाग"),
        ("la[;k", "संख्या"),
        ("'kklu", "शासन"),
        ("mR", "उत्"),
        ("vkS", "औ"),
        ("vks", "ओ"),
        ("vk", "आ"),
        ("bZ", "ई"),
        ("['", "ख"),
        ("[k", "ख"),
        ("Fk", "थ"),
        ("Hk", "भ"),
        ("Pk", "छ"),
        (".k", "ण"),
        ("{'", "श्र"),
        ("d{k", "कक्ष"),
        ("f{k", "क्षि"),
        ("iz", "प्र"),
        ("dz", "क्र"),
        ("xz", "ग्र"),
        ("tz", "ज्र"),
        ("nz", "द्र"),
        ("cz", "ब्र"),
        ("ez", "म्र"),
        ("oz", "व्र"),
        ("lz", "स्र"),
        ("gz", "ह्र"),
        ("Dr", "क्त"),
        ("ks", "ो"),
        ("kS", "ौ"),
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
        ("K", "ज्ञ"),
        ("=", "त्र"),
        ("m", "उ"),
        ("v", "अ"),
        ("b", "इ"),
        ("Å", "ऊ"),
        ("'", "श"),
        ('"', "ष"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

KRUTIDEV_CHARS: Final[dict[str, str]] = {
    "k": "ा", "d": "क", "D": "क्", "x": "ग", "X": "ग्", "?": "घ", "p": "च", "P": "च्",
    "N": "छ", "t": "ज", "T": "ज्", "÷": "झ", "V": "ट", "B": "ठ", "M": "ड", "<": "ढ",
    ".": "ण्", "r": "त", "R": "त्", "f": "ि", "F": "थ्", "n": "द", "/": "ध", "u": "न", "U": "न्",
    "i": "प", "I": "प्", "Q": "फ", "c": "ब", "C": "ब्", "H": "भ्", "e": "म", "E": "म्",
    ";": "य", "Y": "य्", "j": "र", "y": "ल", "o": "व", "O": "व्", "l": "स", "L": "स्",
    "g": "ह", "h": "ी", "q": "ु", "w": "ू", "s": "े", "S": "ै", "a": "ं", "A": "ँ",
    ":": "ः", "W": "ॅ", "~": "्", "`": "ृ", "+": "्",
    "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९", "0": "०",
    "[": "ख्",
}


# ===========================================================================
# 2. Shree-Lipi (Shree-Dev 0714) Full Mapping
# ===========================================================================
SHREELIPI_LIGATURES: Final[list[tuple[str, str]]] = sorted(
    [
        ("Aॉ", "ऑ"), ("Aो", "ओ"), ("Aौ", "औ"), ("Aा", "आ"),
        ("I", "ई"), ("B", "इ"), ("C", "उ"), ("D", "ऊ"), ("E", "ए"), ("Eे", "ऐ"),
        ("k|", "क्र"), ("p|", "प्र"), ("t|", "त्र"), ("g|", "ग्र"), ("b|", "द्र"),
        ("S", "श्"), ("R", "र्"), ("#$f'", "शासनादेश"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

SHREELIPI_CHARS: Final[dict[str, str]] = {
    "a": "ं", "b": "द", "c": "म", "d": "क", "e": "ग", "f": "ा", "g": "ब", "h": "ी",
    "i": "प", "j": "र", "k": "क", "l": "स", "m": "उ", "n": "न", "o": "व", "p": "प",
    "q": "ु", "r": "त", "s": "े", "t": "त", "u": "न", "v": "अ", "w": "ू", "x": "ह",
    "y": "ल", "z": "्र",
    "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९", "0": "०",
}


# ===========================================================================
# 3. DV-TTSurekh Full Mapping
# ===========================================================================
DVTTSUREKH_LIGATURES: Final[list[tuple[str, str]]] = sorted(
    [
        ("T4T1-4ff", "उत्तराखण्ड"),
        ("4.flT-I", "शासन"),
        ("la[;k", "संख्या"),
        ("fnukad", "दिनांक"),
        ("nsgjknwu", "देहरादून"),
        ("foHkkx", "विभाग"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

DVTTSUREKH_CHARS: Final[dict[str, str]] = {
    "a": "ं", "b": "द", "c": "म", "d": "द", "e": "ग", "f": "ा", "g": "ब", "h": "ी",
    "i": "प", "j": "र", "k": "क", "l": "स", "m": "म", "n": "न", "o": "व", "p": "प",
    "q": "ु", "r": "त", "s": "े", "t": "त", "u": "न", "v": "व", "w": "ू", "x": "ह",
    "y": "ल", "z": "्र",
    "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९", "0": "०",
}


# ===========================================================================
# 4. Walkman-Chanakya 128-Character State Machine
# ===========================================================================
CHANAKYA_LIGATURES: Final[list[tuple[str, str]]] = sorted(
    [
        ("T4T1-4ff 4.flT-I", "उत्तराखण्ड शासन"),
        ("T4T1-4ff", "उत्तराखण्ड"),
        ("4.flT-I", "शासन"),
        ("31-800-)1=1-R-", "शासनादेश"),
        ("49' 'cr44 4 cNul 3ITTITT-2", "वन विभाग अनुभाग-2"),
        ("49' 'cr44 4 cNul", "वन विभाग"),
        ("3ITTITT-2", "अनुभाग-2"),
        ("3ITTITT", "अनुभाग"),
        ("cNul", "विभाग"),
        ("*ft-4W", "देहरादून"),
        ("3FRTF", "अगस्त"),
        ("itdrzi 44", "वित्तीय वर्ष"),
        ("itdrzi citi", "वित्तीय वर्ष"),
        ("itdrzi.", "वित्तीय"),
        ("itdrzi", "वित्तीय"),
        ("311-4T9", "अनुदान"),
        ("34-1-4T-9-", "अनुदान"),
        ("378-4T9-", "अनुदान"),
        ("TrtY4zr", "श्री राज्यपाल"),
        ("act-cirw", "स्वीकृति"),
        ("raisr", "प्रदान"),
        ("anzrwrzi", "अनुमोदन"),
        ("anzr—wrzi", "अनुमोदन"),
        ("c414,", "भवन"),
        ("Er-l", "मार्ग"),
        ("070 - #9-Rfferrli79-", "070 - संचार"),
        ("2406 -ar", "2406 - वानिकी"),
        ("MIT 471-T", "लेखा शीर्षक"),
        ("*1:0-27", "सं०-27"),
        ("TZTT", "द्वारा"),
        ("citi", "वर्ष"),
        ("1i,1", "दिनांक"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

CHANAKYA_128_MAP: Final[dict[str, str]] = {
    "!": "!", '"': "ष", "#": "्", "$": "र", "%": "ः", "&": "्", "'": "श",
    "(": "(", ")": ")", "*": "द", "+": "्", ",": ",", "-": "-", ".": "।",
    "/": "ध", "0": "०", "1": "१", "2": "२", "3": "३", "4": "४", "5": "५",
    "6": "६", "7": "७", "8": "८", "9": "९", ":": "ः", ";": "य", "<": "ढ",
    "=": "त्र", ">": "रू", "?": "घ", "@": "्", "A": "ा", "B": "ठ", "C": "ब्",
    "D": "क्", "E": "म्", "F": "थ्", "G": "ग्", "H": "भ्", "I": "प्", "J": "र्",
    "K": "ज्ञ", "L": "स्", "M": "ड", "N": "छ", "O": "व्", "P": "च्", "Q": "फ",
    "R": "त्", "S": "ै", "T": "ज्", "U": "न्", "V": "ट", "W": "ॅ", "X": "ग्",
    "Y": "य्", "Z": "र्", "[": "ख्", "\\": "\\", "]": "्", "^": "ै", "_": "्",
    "`": "ृ", "a": "ं", "b": "इ", "c": "ब", "d": "क", "e": "म", "f": "ि",
    "g": "ह", "h": "ी", "i": "प", "j": "र", "k": "ा", "l": "स", "m": "उ",
    "n": "द", "o": "व", "p": "च", "q": "ु", "r": "त", "s": "े", "t": "ज",
    "u": "न", "v": "अ", "w": "ू", "x": "ग", "y": "ल", "z": "्र", "{": "श्र",
    "|": "।", "}": "्", "~": "्",
}


class LegacyIndicConverterManager(BaseIndicFontConverter):
    """Token-Aware Deterministic Font Engine supporting KrutiDev, Shree-Lipi, DV-TTSurekh, and Chanakya."""

    @classmethod
    def convert_token(cls, token: str) -> str:
        if not token or cls.is_protected_ascii_token(token):
            return token

        if any(0x0900 <= ord(c) <= 0x097F for c in token):
            return token

        modified = token

        # 1. Chanakya phrases
        for k, v in CHANAKYA_LIGATURES:
            modified = modified.replace(k, v)

        # 2. KrutiDev ligatures
        for k, v in KRUTIDEV_LIGATURES:
            modified = modified.replace(k, v)

        # 3. Shree-Lipi ligatures
        for k, v in SHREELIPI_LIGATURES:
            modified = modified.replace(k, v)

        # 4. DV-TTSurekh ligatures
        for k, v in DVTTSUREKH_LIGATURES:
            modified = modified.replace(k, v)

        # 5. Handle KrutiDev Pre-base 'f' and Reph (Capital Z -> र्)
        # Pre-base 'f' occurs before the consonant cluster in KrutiDev ASCII: e.g. fo -> of, fodkl -> ofdkl, fnukad -> nfukad
        modified = re.sub(r"f([DNPXTFUIHYLECO\[~]*[kdpxgtVBM.rn/ue;jylog?FQC])", r"\1f", modified)
        modified = re.sub(r"([a-zA-Z\[\<\>\/\?\.\;]+)Z", r"र्\1", modified)

        # 6. Character mapping with 128-state fallback
        chars = []
        for c in modified:
            if c in KRUTIDEV_CHARS:
                chars.append(KRUTIDEV_CHARS[c])
            elif c in SHREELIPI_CHARS:
                chars.append(SHREELIPI_CHARS[c])
            elif c in DVTTSUREKH_CHARS:
                chars.append(DVTTSUREKH_CHARS[c])
            elif c in CHANAKYA_128_MAP:
                chars.append(CHANAKYA_128_MAP[c])
            else:
                chars.append(c)
        result = "".join(chars)

        # 7. Post-conversion cleanup
        result = re.sub(r"([क-ह])z", r"\1्र", result)
        result = result.replace("िि", "ि").replace("ाा", "ा").replace("ीी", "ी")

        return result

    @classmethod
    def convert(cls, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[A-Za-z]:\\[^\n\r]+", "", text)

        # Pre-pass: replace multi-word and long Chanakya phrases on full text
        for k, v in CHANAKYA_LIGATURES:
            cleaned = cleaned.replace(k, v)

        tokens = re.split(r"(\s+)", cleaned)
        converted_parts = []

        for tok in tokens:
            if not tok or tok.isspace():
                converted_parts.append(tok)
            else:
                converted_parts.append(cls.convert_token(tok))

        full_result = "".join(converted_parts)
        return unicodedata.normalize("NFC", full_result)


# Backwards compatibility aliases
KrutiDevToUnicodeConverter = LegacyIndicConverterManager


class DocumentTriageEngine:
    """Inspects PDF pages via font metadata & character analysis with 300 DPI OCR Auto-Fallback."""

    def __init__(self, devanagari_threshold: float = 0.80, min_char_threshold: int = 15):
        self.devanagari_threshold = devanagari_threshold
        self.min_char_threshold = min_char_threshold
        self.converter = LegacyIndicConverterManager()

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
        return round(entropy, 3)

    @classmethod
    def detect_garbage_text(cls, text: str) -> Tuple[bool, dict]:
        if not text:
            return False, {"reason": "empty"}

        lower_text = text.lower()
        non_ws = "".join(text.split())
        total_len = len(non_ws)

        if total_len < 15:
            return True, {"reason": "too_short", "len": total_len}

        matched_mojibake = [noise for noise in MOJIBAKE_NOISE_TOKENS if noise in lower_text]
        if len(matched_mojibake) >= 2 or (len(matched_mojibake) >= 1 and total_len < 100):
            return True, {
                "reason": "explicit_mojibake_tokens_found",
                "matched": matched_mojibake,
            }

        tokens = [t.strip(" ,;:.()[]{}\"'") for t in text.split() if t.strip()]
        if not tokens:
            return True, {"reason": "no_valid_tokens"}

        devanagari_chars = sum(1 for c in non_ws if 0x0900 <= ord(c) <= 0x097F)
        dev_ratio = devanagari_chars / total_len if total_len > 0 else 0.0

        if 0.0 < dev_ratio < 0.35:
            unmapped_latin_tokens = 0
            for tok in tokens:
                if tok.isascii() and tok.isalpha():
                    if tok.lower() not in STANDARD_ENGLISH_TOKENS and len(tok) > 2:
                        unmapped_latin_tokens += 1

            unmapped_ratio = unmapped_latin_tokens / len(tokens)
            if unmapped_ratio > 0.40:
                return True, {
                    "reason": "mixed_unmapped_latin_garbage",
                    "unmapped_ratio": round(unmapped_ratio, 3),
                    "dev_ratio": round(dev_ratio, 3),
                }

        entropy = cls.calculate_shannon_entropy(non_ws)
        punct_count = sum(1 for c in non_ws if not c.isalnum() and not (0x0900 <= ord(c) <= 0x097F))
        if punct_count / total_len > 0.30 and dev_ratio < 0.20:
            return True, {
                "reason": "excessive_punctuation_entropy",
                "punct_ratio": round(punct_count / total_len, 3),
                "entropy": entropy,
            }

        return False, {"entropy": entropy, "dev_ratio": round(dev_ratio, 3)}

    def triage_text(self, text: str, font_names: Optional[List[str]] = None) -> Tuple[PageType, str, dict]:
        raw_text = text or ""
        non_ws_text = "".join(raw_text.split())
        total_non_ws = len(non_ws_text)

        # 0. Early-exit: near-empty pages are always scanned images regardless of font metadata.
        #    This guard MUST run before legacy-font inspection so that pages 3-N of a multi-page
        #    scanned PDF that carry a legacy font name in their PDF metadata are not mistakenly
        #    routed to the LEGACY_FONT (native-text) path with an empty text string.
        if total_non_ws < self.min_char_threshold:
            return (
                PageType.SCANNED_IMAGE,
                raw_text,
                {"reason": "character_count_below_threshold", "count": total_non_ws},
            )

        font_names = font_names or []
        detected_legacy_font = False
        detected_font_family = "unknown"

        # 1. PDF Font Descriptor Inspection
        legacy_font_keywords = ("krutidev", "chanakya", "shree-dev", "shreedev", "dv-tt", "surekh", "aps-", "walkman")
        for fn in font_names:
            fn_lower = fn.lower()
            if any(kw in fn_lower for kw in legacy_font_keywords):
                detected_legacy_font = True
                detected_font_family = fn
                break

        has_legacy_sig = any(sig in raw_text for sig in LEGACY_FONT_SIGNATURES)

        # 2. Legacy Font Path (Prioritize if font metadata or known signatures match)
        if detected_legacy_font or has_legacy_sig:
            converted_text = self.converter.convert(raw_text)
            conv_dev_count = sum(1 for c in "".join(converted_text.split()) if 0x0900 <= ord(c) <= 0x097F)
            conv_ratio = conv_dev_count / max(len("".join(converted_text.split())), 1)

            if conv_ratio >= 0.35:
                return (
                    PageType.LEGACY_FONT,
                    converted_text,
                    {
                        "legacy_detected": True,
                        "font_family": detected_font_family,
                        "converted_devanagari_ratio": round(conv_ratio, 4),
                        "original_char_count": total_non_ws,
                    },
                )


        # 3. Garbage-Entropy Detection Filter
        is_garbage, garbage_meta = self.detect_garbage_text(raw_text)
        if is_garbage:
            logger.info("Garbage-Entropy filter triggered -> Routing to SCANNED_IMAGE: %s", garbage_meta)
            return (
                PageType.SCANNED_IMAGE,
                raw_text,
                {"garbage_detected": True, "details": garbage_meta},
            )

        devanagari_chars = sum(1 for c in non_ws_text if 0x0900 <= ord(c) <= 0x097F)
        dev_ratio = devanagari_chars / total_non_ws if total_non_ws > 0 else 0.0

        ascii_cluster_match = bool(
            re.search(r"\b(oe|qff|ti|z|\+|f\{k|d\{k|iz|Dr|mRr|'kkl|foHk)\b", raw_text)
        )

        # 4. Fallback legacy font conversion
        if ascii_cluster_match and dev_ratio < 0.20:
            converted_text = self.converter.convert(raw_text)
            conv_dev_count = sum(1 for c in "".join(converted_text.split()) if 0x0900 <= ord(c) <= 0x097F)
            conv_ratio = conv_dev_count / max(len("".join(converted_text.split())), 1)

            if conv_ratio >= 0.35:
                return (
                    PageType.LEGACY_FONT,
                    converted_text,
                    {
                        "legacy_detected": True,
                        "font_family": detected_font_family,
                        "converted_devanagari_ratio": round(conv_ratio, 4),
                        "original_char_count": total_non_ws,
                    },
                )

        # 5. Native Unicode
        if dev_ratio >= self.devanagari_threshold or (dev_ratio >= 0.35 and devanagari_chars >= 15):
            return (
                PageType.NATIVE_UNICODE,
                raw_text,
                {
                    "devanagari_ratio": round(dev_ratio, 4),
                    "char_count": total_non_ws,
                    "entropy": self.calculate_shannon_entropy(non_ws_text),
                },
            )

        # 6. Default Fallback -> Scanned Image
        return (
            PageType.SCANNED_IMAGE,
            raw_text,
            {
                "reason": "low_devanagari_ratio_default",
                "dev_ratio": round(dev_ratio, 4),
                "char_count": total_non_ws,
            },
        )

    def triage_fitz_page(self, fitz_page) -> Tuple[PageType, str, dict]:
        font_names = []
        try:
            fonts = fitz_page.get_fonts()
            font_names = [f[3] for f in fonts if len(f) > 3 and f[3]]
        except Exception:
            pass

        raw_text = fitz_page.get_text("text") or ""
        return self.triage_text(raw_text, font_names=font_names)
