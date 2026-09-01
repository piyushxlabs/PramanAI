"""Stage 4: Devanagari Unicode Normalizer & Domain Lexicon.

Eliminates Unicode composition mismatches and OCR artifacts.
Performs NFC normalization, distinct handling for U+0970 (Devanagari Abbreviation Sign)
vs U+0966 (Devanagari Zero), canonical Nuqta/Halant ordering repair, contextual ZWJ/ZWNJ preservation,
Nukta-aware pre-base matra reordering, misplaced 'ि' (\u093F) phoneme repair, and comprehensive
Uttarakhand administrative governance glossary corrections.
"""

import logging
import unicodedata
import regex as re

logger = logging.getLogger("gov_pdf_extractor.normalizer")


class DevanagariNormalizer:
    """Standardizes Devanagari Unicode strings and repairs OCR corruptions in administrative texts."""

    ADMIN_OCR_REPAIR_MAP: dict[str, str] = {
        # Administrative Terms & Headings
        r"\bकार्यालय[- ]?झाप\b": "कार्यालय-ज्ञाप",
        r"\bचाजस्व\s+विमान\b": "राजस्व विभाग",
        r"\bकाब्द\s+करें\b": "कष्ट करें",
        r"\bभाव्यांसे\b": "माध्यम से",
        r"\bभुजे\s+यह\b": "मुझे यह",
        r"\bसिंधोचगड\b": "पिथौरागढ़",
        r"\bफरिोरागढ़\b": "पिथौरागढ़",
        r"\bशैक्षांक\b": "शैक्षणिक",
        r"\b(?:उल्कृष्?ष्टता|उत्कृष्?ष्ठता|उल्कृष्टता|उत्कृष्टता|उत्कृष्ठता|उल्कृष्ठता)\b": "उत्कृष्टता",
        r"\bएत्दद्वारा\b": "एतद्द्वारा",

        # Officer Names & Entities
        r"\bकौस्तुम\b": "कौस्तुभ",
        r"\bगेखुरी\b": "मैखुरी",

        # Abbreviation Dots Repair (Devanagari Abbreviation Sign \u0970)
        r"\bएओको\b": "ए०के०",
        r"\bए०\s*को\b": "ए०के०",
        r"\bजेओपी\b": "जे०पी०",
        r"\bजे०\s*पी\b": "जे०पी०",
        r"\bडॉओ\b": "डॉ०",
        r"\bकैओ\b": "कै०",
    }

    ADMIN_OCR_REPAIR_RULES: list[tuple[re.Pattern, str]] = [
        (re.compile(pattern, re.UNICODE), replacement)
        for pattern, replacement in ADMIN_OCR_REPAIR_MAP.items()
    ]

    # Domain-specific lookup and repair mapping for standard Uttarakhand administrative terms
    ADMIN_GLOSSARY_CORRECTIONS: list[tuple[re.Pattern, str]] = [
        # --- Critical Matra Misplacement & Administrative Terms ---
        (re.compile(r"\bदनिांक\b|\bदिनाक\b|\bदीनांक\b|\bदनांक\b", re.UNICODE), "दिनांक"),
        (re.compile(r"\bअधकिारी\b|\bअधीकारी\b|\bअधिकारि\b", re.UNICODE), "अधिकारी"),
        (re.compile(r"\bकर्मिक\b|\bकार्मकि\b|\bकार्मीक\b", re.UNICODE), "कार्मिक"),
        (re.compile(r"\bअलुम्गा\b|\bअनुभग\b|\bअनभाग\b", re.UNICODE), "अनुभाग"),
        (re.compile(r"\bवज्ञिप्ति\b|\bवज्ञप्ति\b|\bविज्ञप्ती\b|\bविज्ञपती\b", re.UNICODE), "विज्ञप्ति"),
        (re.compile(r"\bप्रशक्षिण\b|\bप्रशिक्षन\b|\bप्रशीक्षण\b", re.UNICODE), "प्रशिक्षण"),
        (re.compile(r"\bपथिौरागढ़\b|\bपिथोरागढ़\b|\bपिथौरागढ\b", re.UNICODE), "पिथौरागढ़"),
        (re.compile(r"\bसंस्ततुि\b|\bसंस्तिति\b|\bसंस्तति\b|\bसस्तुति\b|\bसंस्तुती\b", re.UNICODE), "संस्तुति"),
        (re.compile(r"\bवभिाग\b|\bविभग\b|\bवीभाग\b", re.UNICODE), "विभाग"),
        (re.compile(r"\bसचवि\b|\bसचीव\b|\bसचिब\b", re.UNICODE), "सचिव"),
        (re.compile(r"\bनयिम\b|\bनीयम\b", re.UNICODE), "नियम"),
        (re.compile(r"\bस्थति\b|\bस्थीति\b|\bस्थिती\b", re.UNICODE), "स्थिति"),
        (re.compile(r"\bसमति\b|\bसमीति\b|\bसमिती\b", re.UNICODE), "समिति"),
        (re.compile(r"\bप्रतलिपिि\b|\bप्रतलिपि\b|\bप्रतिलिपी\b|\bप्रतीलिपि\b", re.UNICODE), "प्रतिलिपि"),
        (re.compile(r"\bअधसूिचति\b|\bअधिसूचीत\b|\bअधिसूचित\b", re.UNICODE), "अधिसूचित"),
        (re.compile(r"\bअधसूिचना\b|\bअधिसुचना\b|\bअधीसूचना\b", re.UNICODE), "अधिसूचना"),
        (re.compile(r"\bनर्दिेश\b|\bनीर्देश\b|\bनिरदेश\b", re.UNICODE), "निर्देश"),
        (re.compile(r"\bनर्दिेशानुसार\b|\bनिर्देशा\s*नुसार\b", re.UNICODE), "निर्देशानुसार"),
        (re.compile(r"\bपत्रावलि\b|\bपत्रावली\b", re.UNICODE), "पत्रावली"),

        # --- Core Government & Executive ---
        (re.compile(r"शास(?:ना|ा|न)?देश|शास\s*नादेश|शासना\s*देश|शाशनादेश|शासनादश|शासनादेष", re.UNICODE), "शासनादेश"),
        (re.compile(r"उत्तराख(?:ण्ड|न्ड|ंड)\s*शासन|उत्तराखण्\s*ड\s*शासन", re.UNICODE), "उत्तराखण्ड शासन"),
        (re.compile(r"मुख्य\s*मंत्री|मुख्यमंत्री|मुख्यमत्री", re.UNICODE), "मुख्यमंत्री"),
        (re.compile(r"मुख्य\s*सचिव|मुख्यसचिव|मुख्य\s*सचीव|मुख्य\s*सचवि", re.UNICODE), "मुख्य सचिव"),
        (re.compile(r"अपर\s*मुख्य\s*सचिव|अपरमुख्यसचिव|अपर\s*मुख्य\s*सचवि", re.UNICODE), "अपर मुख्य सचिव"),
        (re.compile(r"प्रमुख\s*सचिव|प्रमुखसचिव|प्रमुख\s*सचीव|प्रमुख\s*सचवि", re.UNICODE), "प्रमुख सचिव"),
        (re.compile(r"अपर\s*सचिव|अपरसचिव|अपर\s*सचवि", re.UNICODE), "अपर सचिव"),
        (re.compile(r"संयुक्त\s*सचिव|सयुक्त\s*सचिव|संयुक्त\s*सचवि", re.UNICODE), "संयुक्त सचिव"),
        (re.compile(r"उप\s*सचिव|उपसचिव|उप\s*सचवि", re.UNICODE), "उप सचिव"),
        (re.compile(r"अनु\s*सचिव|अनुसचिव|अनु\s*सचवि", re.UNICODE), "अनु सचिव"),
        (re.compile(r"अनुभाग\s*अधिकारी|अनुभागाधिकारी|अनुभाग\s*अधीकारी|अनुभाग\s*अधकिारी", re.UNICODE), "अनुभाग अधिकारी"),
        (re.compile(r"कार्या\s*लय|कार्यालय|कारयालय|कायालय", re.UNICODE), "कार्यालय"),
        (re.compile(r"सचिवा\s*लय|सचिवालय|सचीवालय", re.UNICODE), "सचिवालय"),
        (re.compile(r"निदेशा\s*लय|निदेशालय|निदेशलय", re.UNICODE), "निदेशालय"),
        (re.compile(r"महानिदेशा\s*लय|महानिदेशालय", re.UNICODE), "महानिदेशालय"),
        (re.compile(r"आयुक्त|आयक्त", re.UNICODE), "आयुक्त"),
        (re.compile(r"जिला\s*धिकारी|जिलाधिकारी|जलिाधिकारी|जलिाधकिारी", re.UNICODE), "जिलाधिकारी"),
        (re.compile(r"मुख्य\s*विकास\s*अधिकारी|मुख्यविकासअधिकारी|मुख्य\s*वकिास\s*अधकिारी", re.UNICODE), "मुख्य विकास अधिकारी"),
        (re.compile(r"उप\s*जिलाधिकारी|उपजिलाधिकारी|उपजलिाधकिारी", re.UNICODE), "उप जिलाधिकारी"),
        (re.compile(r"तहसील\s*दार|तहसीलदार|तहसिलदार", re.UNICODE), "तहसीलदार"),

        (re.compile(r"\bवत्ति\b", re.UNICODE), "वित्त"),
        (re.compile(r"(?:वित्त|वत्ति)\s*(?:विभाग|वभिाग)", re.UNICODE), "वित्त विभाग"),
        (re.compile(r"(?:वित्त|वत्ति)\s*(?:अनुभाग)", re.UNICODE), "वित्त अनुभाग"),
        (re.compile(r"(?:कार्मिक|कार्मकि|कर्मिक)\s*(?:विभाग|वभिाग)", re.UNICODE), "कार्मिक विभाग"),
        (re.compile(r"(?:कार्मिक|कार्मकि|कर्मिक)\s*(?:अनुभाग|अलुम्गा)", re.UNICODE), "कार्मिक अनुभाग"),
        (re.compile(r"राजस्व\s*(?:विभाग|वभिाग)", re.UNICODE), "राजस्व विभाग"),
        (re.compile(r"गृह\s*(?:विभाग|वभिाग)", re.UNICODE), "गृह विभाग"),
        (re.compile(r"चिकित्सा\s*स्वास्थ्य\s*एवं\s*परिवार\s*कल्याण|चिकित्सा\s*विभाग|चकित्सा\s*वभिाग", re.UNICODE), "चिकित्सा स्वास्थ्य एवं परिवार कल्याण"),
        (re.compile(r"लोक\s*निर्माण\s*विभाग|लो\s*नि\s*वि|लो०नि०वि०", re.UNICODE), "लोक निर्माण विभाग"),
        (re.compile(r"सूचना\s*प्रौद्योगिकी\s*विकास\s*एजेंसी|आईटीडीए|आई०टी०डी०ए०", re.UNICODE), "सूचना प्रौद्योगिकी विकास एजेंसी"),

        # --- Financial & Budget Terminology ---
        (re.compile(r"अवमु\s*क्त|अवमक्त|अवमुकत|अवमुक्त", re.UNICODE), "अवमुक्त"),
        (re.compile(r"स्वी\s*कृति|स्वीकृति|स्वीक्रती|स्वीकृती|स्वीकति", re.UNICODE), "स्वीकृति"),
        (re.compile(r"अनु\s*दान\s*(?:संख्या|सं[०॰o\.]|सं\.)|अनुदान\s*सख्या|अनुदान\s*संख्या", re.UNICODE), "अनुदान संख्या"),
        (re.compile(r"लेखा\s*शीर्ष|लेखाशीर्ष|लेखाशीष|लेखा\s*शीर्षक|लेखाशिर्ष", re.UNICODE), "लेखाशीर्ष"),
        (re.compile(r"मुख्य\s*शीर्ष|मुख्यशीर्ष", re.UNICODE), "मुख्य शीर्ष"),
        (re.compile(r"उप\s*मुख्य\s*शीर्ष|उपमुख्यशीर्ष", re.UNICODE), "उप मुख्य शीर्ष"),
        (re.compile(r"लघु\s*शीर्ष|लघुशीर्ष", re.UNICODE), "लघु शीर्ष"),
        (re.compile(r"उप\s*शीर्ष|उपशीर्ष", re.UNICODE), "उप शीर्ष"),
        (re.compile(r"विस्तृत\s*शीर्ष|विस्तृतशीर्ष", re.UNICODE), "विस्तृत शीर्ष"),
        (re.compile(r"उद्देश्य\s*शीर्ष|उद्देश्यशीर्ष", re.UNICODE), "उद्देश्य शीर्ष"),
        (re.compile(r"पुन\s*रीक्षित|पुनरीक्षित|पुनरिक्षित", re.UNICODE), "पुनरीक्षित"),
        (re.compile(r"आय\s*व्ययक|आयव्ययक|आयवययक", re.UNICODE), "आयव्ययक"),
        (re.compile(r"व्यय\s*भार|व्ययभार", re.UNICODE), "व्ययभार"),
        (re.compile(r"धन\s*राशि|धनराशि|धनराशी", re.UNICODE), "धनराशि"),
        (re.compile(r"राज्यांश|राज्य\s*अंश|राज्याश", re.UNICODE), "राज्यांश"),
        (re.compile(r"केन्द्रांश|केन्द्र\s*अंश|केन्द्राश", re.UNICODE), "केन्द्रांश"),
        (re.compile(r"आकस्मिकता\s*निधि|आकस्मिकतानिधि", re.UNICODE), "आकस्मिकता निधि"),
        (re.compile(r"संचित\s*निधि|संचितनिधि", re.UNICODE), "संचित निधि"),
        (re.compile(r"कोषा\s*गार|कोषागार|कोषागाार", re.UNICODE), "कोषागार"),
        (re.compile(r"उप\s*कोषागार|उपकोषागार", re.UNICODE), "उप कोषागार"),
        (re.compile(r"कोषाधिकारी|कोषा\s*धिकारी|कोषाधकिारी", re.UNICODE), "कोषाधिकारी"),
        (re.compile(r"महालेखाकार|महा\s*लेखाकार", re.UNICODE), "महालेखाकार"),
        (re.compile(r"उपयोगिता\s*प्रमाण\s*पत्र|उपयोगिता\s*प्रमाण-पत्र|उपयोगिता\s*प्रमाणपत्र", re.UNICODE), "उपयोगिता प्रमाण-पत्र"),
        (re.compile(r"ऋण\s*एवं\s*अग्रिम|ऋण\s*तथा\s*अग्रिम", re.UNICODE), "ऋण एवं अग्रिम"),
        (re.compile(r"पूंजीगत\s*परिव्यय|पूंजीगत\s*व्यय", re.UNICODE), "पूंजीगत परिव्यय"),
        (re.compile(r"प्रत्याहरण|धनराशि\s*प्रत्याहरण", re.UNICODE), "प्रत्याहरण"),
        (re.compile(r"पुनर्विनियोग|पुनःविनियोग", re.UNICODE), "पुनर्विनियोग"),

        # --- Document Types, Identifiers & Standard Clauses ---
        (re.compile(r"\bसख्या\b|\bसंख्\s*या\b", re.UNICODE), "संख्या"),
        (re.compile(r"कार्यालय\s*ज्ञाप|कार्यालय\s*ज्ञापन", re.UNICODE), "कार्यालय ज्ञाप"),
        (re.compile(r"परिपत्र|परि-पत्र", re.UNICODE), "परिपत्र"),
        (re.compile(r"पत्रांक\s*[:\-]?|पत्राक", re.UNICODE), "पत्रांक"),
        (re.compile(r"देहरा\s*दून|देहरादून|देहरादुन", re.UNICODE), "देहरादून"),
        (re.compile(r"अनु\s*भाग|अनुभाग", re.UNICODE), "अनुभाग"),
        (re.compile(r"संलग्नक|सलग्नक", re.UNICODE), "संलग्नक"),
        (re.compile(r"परिशिष्ट|परिशिष्ठ", re.UNICODE), "परिशिष्ट"),
        (re.compile(r"प्रतिलिपि\s*निम्नांकित\s*को\s*सूचनार्थ|प्रतिलिपि", re.UNICODE), "प्रतिलिपि"),
        (re.compile(r"आज्ञा\s*से|आज्ञासे", re.UNICODE), "आज्ञा से"),
        (re.compile(r"भवदीय|भवदीया", re.UNICODE), "भवदीय"),
        (re.compile(r"हस्ताक्षर|हस्ताक्षरित", re.UNICODE), "हस्ताक्षर"),
        (re.compile(r"समसंख्यक|सम\s*संख्यक", re.UNICODE), "समसंख्यक"),
        (re.compile(r"तद्दिनांक|तद\s*दिनांक", re.UNICODE), "तद्दिनांक"),
        (re.compile(r"एतद्द्वारा|एतद\s*द्वारा", re.UNICODE), "एतद्द्वारा"),
        (re.compile(r"सर्वसाधारण|सर्व\s*साधारण", re.UNICODE), "सर्वसाधारण"),
        (re.compile(r"प्रख्यापित\s*किया\s*जाता\s*है", re.UNICODE), "प्रख्यापित किया जाता है"),
    ]

    # Devanagari abbreviation rules: map abbreviation markers following acronym initials to U+0970 (॰)
    ABBREVIATION_SIGN_RULES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"(?<=[\s\p{P}]|^)सं[०o\.]", re.UNICODE), "सं॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)दि[०o\.]", re.UNICODE), "दि॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)पं[०o\.]", re.UNICODE), "पं॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)क्र[०o\.]", re.UNICODE), "क्र॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)रु[०o\.]", re.UNICODE), "रु॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)ले[०o\.]", re.UNICODE), "ले॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)अ[०o\.]\s*शा[०o\.]", re.UNICODE), "अ॰शा॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)शा[०o\.]\s*आ[०o\.]", re.UNICODE), "शा॰आ॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)डा[०o\.]", re.UNICODE), "डा॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)क्रम\s*सं[०o\.]", re.UNICODE), "क्रम सं॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)पृ[०o\.]", re.UNICODE), "पृ॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)प्र[०o\.]\s*मु[०o\.]\s*व[०o\.]\s*सं[०o\.]", re.UNICODE), "प्र॰मु॰व॰सं॰"),
        (re.compile(r"(?<=[\s\p{P}]|^)लो[०o\.]\s*नि[०o\.]\s*वि[०o\.]", re.UNICODE), "लो॰नि॰वि॰"),
    ]

    # OCR confusion rules (numeral zero vs letter O)
    OCR_CONFUSION_RULES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\bकम\s*(?:सं[०॰o\.]|संख्या|\.)", re.UNICODE), "क्रम सं॰"),
        (re.compile(r"\bकम\s*(\d+)", re.UNICODE), r"क्रम \1"),
        (re.compile(r"(?<=\d|[\u0966-\u096F])O(?=\d|[\u0966-\u096F])", re.UNICODE), "०"),
        (re.compile(r"(?<=\p{Devanagari})\s*O\s*(?=\p{Devanagari}|\d)", re.UNICODE), "०"),
        (re.compile(r"(?<=\p{Devanagari})\s*l\s*(?=\d)", re.UNICODE), "1"),
    ]

    @classmethod
    def normalize_nfc(cls, text: str) -> str:
        """Applies Unicode Standard NFC normalization."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @classmethod
    def repair_nuqta_and_halant_ordering(cls, text: str) -> str:
        """Enforces canonical Devanagari ordering: Base Consonant + Nuqta + Halant/Matra.

        Fixes inverted OCR ordering (e.g. Consonant + Halant + Nuqta -> Consonant + Nuqta + Halant),
        removes dangling/orphaned Nuqtas (़ U+093C) and Halants (् U+094D) lacking a base consonant.
        """
        if not text:
            return ""

        # 1. Fix inverted Halant + Nuqta: [Consonant] + ् + ़  -->  [Consonant] + ़ + ्
        cleaned = re.sub(r"([\u0915-\u0939])\u094D\u093C", r"\1\u093C\u094D", text)

        # 2. Fix inverted Matra + Nuqta: [Consonant] + [Matra] + ़  -->  [Consonant] + ़ + [Matra]
        cleaned = re.sub(r"([\u0915-\u0939])([\u093E-\u094C\u0962\u0963])\u093C", r"\1\u093C\2", cleaned)

        # 3. Strip orphaned/dangling Nuqta (़ \u093C) at start of string or following whitespace/punctuation
        cleaned = re.sub(r"(?:^|(?<=[\s\p{P}]))\u093C+", "", cleaned)

        # 4. Strip orphaned/dangling Halant (् \u094D) at start of string or following whitespace/punctuation
        cleaned = re.sub(r"(?:^|(?<=[\s\p{P}]))\u094D+", "", cleaned)

        # 5. Deduplicate consecutive Nuqtas
        cleaned = re.sub(r"\u093C{2,}", "\u093C", cleaned)

        return cleaned

    @classmethod
    def repair_matras_and_zwj(cls, text: str) -> str:
        """Fixes matras, Nukta-aware pre-base 'i' matra reordering, and contextually preserves ZWJ after halant.

        Preserves: Halant + ZWJ (्\u200D) for legitimate half-forms (क्, थ्, श्).
        Strips: Isolated ZWJ and unconditional ZWNJ (\u200C).
        """
        if not text:
            return ""

        # 1. Clean duplicate consecutive matras FIRST (e.g. 'कार्मििक' -> 'कार्मिक', 'नििदेशक' -> 'निदेशक')
        cleaned = re.sub(r"(\u093E)+", r"\1", text)  # double aa 'ा'
        cleaned = re.sub(r"(\u093F)+", r"\1", cleaned)  # double chhoti i 'ि'
        cleaned = re.sub(r"(\u0940)+", r"\1", cleaned)  # double badi i 'ी'
        cleaned = re.sub(r"(\u0941)+", r"\1", cleaned)  # double u 'ु'
        cleaned = re.sub(r"(\u0942)+", r"\1", cleaned)  # double oo 'ू'
        cleaned = re.sub(r"(\u0947)+", r"\1", cleaned)  # double e 'े'
        cleaned = re.sub(r"(\u0948)+", r"\1", cleaned)  # double ai 'ै'
        cleaned = re.sub(r"(\u094B)+", r"\1", cleaned)  # double o 'ो'
        cleaned = re.sub(r"(\u094C)+", r"\1", cleaned)  # double au 'ौ'
        cleaned = re.sub(r"(\u0902)+", r"\1", cleaned)  # double anusvara 'ं'

        # 2. Remove ZWNJ unconditionally as it prevents legitimate Hindi conjunct formation
        cleaned = cleaned.replace("\u200C", "")

        # 3. Contextual ZWJ handling: strip ZWJ ONLY if not preceded by a halant
        cleaned = re.sub(r"(?<!\u094D)\u200D+", "", cleaned)

        # 4. Nukta-Aware pre-base chhoti 'i' matra reordering:
        # Matches chhoti 'i' (\u093F) misplaced before consonant clusters (with optional Nuqta and Halants)
        # when NOT already preceded by a base consonant
        # e.g., ि + व -> वि, ि + क + ् + ष -> क्षि, ि + क + ़ + ् + ष -> क़्षि
        cleaned = re.sub(
            r"(?<![\u0904-\u0939]\u093C?)(\u093F)((?:[\u0904-\u0939]\u093C?\u094D)*[\u0904-\u0939]\u093C?)",
            r"\2\1",
            cleaned,
        )

        # 5. Fix dangling matras at start of line or following whitespace/punctuation
        cleaned = re.sub(r"(?:^|(?<=[\s\p{P}]))[\u093E-\u094C\u0962\u0963\u0902\u0903]+", "", cleaned)

        # 6. Repair broken conjunct spaces (e.g. प ् र -> प्र, क ् त -> क्त)
        cleaned = re.sub(r"([\u0915-\u0939]\u093C?)\s*्\s*([\u0915-\u0939]\u093C?)", r"\1्\2", cleaned)

        return cleaned

    @classmethod
    def repair_misplaced_phonemes(cls, text: str) -> str:
        """Repairs common OCR / VLM phoneme inversion where \u093f or matras attach to subsequent consonant."""
        if not text:
            return ""

        result = text
        # Patterns where matra was placed on following consonant instead of leading consonant
        # e.g. दनिांक -> दिनांक (द + न + ि + ा + ं + क -> द + ि + न + ा + ं + क)
        result = re.sub(r"\bदनिांक\b", "दिनांक", result)
        result = re.sub(r"\bअधकिारी\b", "अधिकारी", result)
        result = re.sub(r"\bकार्मकि\b|\bकार्मििक\b", "कार्मिक", result)
        result = re.sub(r"\bवज्ञिप्ति\b", "विज्ञप्ति", result)
        result = re.sub(r"\bप्रशक्षिण\b", "प्रशिक्षण", result)
        result = re.sub(r"\bपथिौरागढ़\b", "पिथौरागढ़", result)
        result = re.sub(r"\bसंस्ततुि\b", "संस्तुति", result)
        result = re.sub(r"\bसामूहकि\b", "सामूहिक", result)
        result = re.sub(r"\bव्यक्तगित\b", "व्यक्तिगत", result)
        result = re.sub(r"\bवभिाग\b", "विभाग", result)
        result = re.sub(r"\bसचवि\b", "सचिव", result)
        result = re.sub(r"\bनयिम\b", "नियम", result)
        result = re.sub(r"\bअधशिासी\b|\bअधशिशी\b", "अधिशासी", result)
        result = re.sub(r"\bअभयिन्ता\b", "अभियन्ता", result)
        result = re.sub(r"\bनतिकिा\b", "नितिका", result)
        result = re.sub(r"\bराधकिा\b", "राधिका", result)
        result = re.sub(r"\bमोनकिा\b", "मोनिका", result)
        result = re.sub(r"\bतविारी\b", "तिवारी", result)
        result = re.sub(r"\bऋषकिेश\b", "ऋषिकेश", result)
        result = re.sub(r"\bनदिेशक\b", "निदेशक", result)
        result = re.sub(r"\bजलिाधकिारी\b", "जिलाधिकारी", result)
        result = re.sub(r"\bकर्मचारयिों\b", "कर्मचारियों", result)
        result = re.sub(r"\bअधकिारयिों\b", "अधिकारियों", result)

        return result

    @classmethod
    def apply_abbreviation_and_glossary_rules(cls, text: str) -> str:
        """Applies distinct Devanagari abbreviation sign (U+0970) rules and domain glossary corrections."""
        if not text:
            return ""

        result = text

        # Apply phoneme repairs first
        result = cls.repair_misplaced_phonemes(result)

        # Apply abbreviation sign rules first (distinguishing U+0970 from numeric zero)
        for pattern, replacement in cls.ABBREVIATION_SIGN_RULES:
            result = pattern.sub(replacement, result)

        # Apply administrative glossary corrections
        for pattern, replacement in cls.ADMIN_GLOSSARY_CORRECTIONS:
            result = pattern.sub(replacement, result)

        # Apply specific administrative OCR repair map
        for pattern, replacement in cls.ADMIN_OCR_REPAIR_RULES:
            result = pattern.sub(replacement, result)

        # Apply OCR confusion rules
        for pattern, replacement in cls.OCR_CONFUSION_RULES:
            result = pattern.sub(replacement, result)

        # Normalize multiple spaces
        result = re.sub(r"[ \t]+", " ", result)

        return result.strip()

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Full Stage 4 normalizer: NFC, Nuqta/Halant ordering, Matra/ZWJ repair, U+0970 abbreviation & glossary."""
        if not text:
            return ""

        # Step 1: NFC standard normalization
        t1 = cls.normalize_nfc(text)

        # Step 2: Nuqta and Halant ordering repair
        t2 = cls.repair_nuqta_and_halant_ordering(t1)

        # Step 3: Matra and ZWJ/ZWNJ repair
        t3 = cls.repair_matras_and_zwj(t2)

        # Step 4: U+0970 Abbreviation Sign and glossary corrections
        t4 = cls.apply_abbreviation_and_glossary_rules(t3)

        # Final NFC pass to guarantee canonical Unicode composition
        return unicodedata.normalize("NFC", t4)


ADMIN_OCR_REPAIR_MAP = DevanagariNormalizer.ADMIN_OCR_REPAIR_MAP


def normalize_devanagari_text(text: str) -> str:
    """Convenience functional wrapper for DevanagariNormalizer.normalize_text."""
    return DevanagariNormalizer.normalize_text(text)


__all__ = ["DevanagariNormalizer", "normalize_devanagari_text", "ADMIN_OCR_REPAIR_MAP"]
