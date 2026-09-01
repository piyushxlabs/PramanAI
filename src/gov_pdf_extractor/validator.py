"""Stage 5: 2D Constraint-Based Mathematical Validation & Multi-Variable Reconciliation Loop.

Performs deterministic arithmetic validation across financial budget tables and government
order allocations. Implements Indian currency sanitization (parenthesized negatives, lakhs/crores formatting),
2D vertical sum and horizontal balance constraints, and multi-variable self-healing using an
OCR digit confusion penalty matrix.
"""

from decimal import Decimal, InvalidOperation
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import regex as re

from src.gov_pdf_extractor.models import BoundingBox, TableCell, TableData

logger = logging.getLogger("gov_pdf_extractor.validator")

# Devanagari digit to ASCII digit translation map
DEV_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Indian scale multiplier words
INDIAN_SCALE_MULTIPLIERS = {
    "करोड़": Decimal("10000000"),
    "करोड": Decimal("10000000"),
    "cr": Decimal("10000000"),
    "crore": Decimal("10000000"),
    "crores": Decimal("10000000"),
    "लाख": Decimal("100000"),
    "lakh": Decimal("100000"),
    "lacs": Decimal("100000"),
    "हजार": Decimal("1000"),
    "हज़ार": Decimal("1000"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
}

# Header scale detection patterns
HEADER_SCALE_PATTERNS: list[tuple[re.Pattern, Decimal, str]] = [
    (re.compile(r"(?:धनराशि|राशि|आवंटन|व्यय|रुपये|रु०|रु\.|rs\.?|inr)?\s*(?:in|\(|में|/|\:)?\s*(?:करोड़|करोड|crores?|cr)\b", re.IGNORECASE | re.UNICODE), Decimal("10000000"), "करोड़"),
    (re.compile(r"(?:धनराशि|राशि|आवंटन|व्यय|रुपये|रु०|रु\.|rs\.?|inr)?\s*(?:in|\(|में|/|\:)?\s*(?:लाख|lakhs?|lacs?)\b", re.IGNORECASE | re.UNICODE), Decimal("100000"), "लाख"),
    (re.compile(r"(?:धनराशि|राशि|आवंटन|व्यय|रुपये|रु०|रु\.|rs\.?|inr)?\s*(?:in|\(|में|/|\:)?\s*(?:हजार|हज़ार|thousands?|k)\b", re.IGNORECASE | re.UNICODE), Decimal("1000"), "हज़ार"),
]

# OCR Digit Confusion Pairs (digit -> possible misread alternatives, cost)
DIGIT_CONFUSION_MAP: dict[str, list[tuple[str, float]]] = {
    "3": [("8", 1.0), ("9", 1.5), ("2", 2.0)],
    "8": [("3", 1.0), ("0", 1.2), ("9", 1.5)],
    "1": [("7", 1.0), ("4", 1.8), ("l", 1.0), ("I", 1.0)],
    "7": [("1", 1.0), ("9", 1.8)],
    "0": [("8", 1.2), ("6", 1.8), ("O", 0.5), ("o", 0.5)],
    "5": [("6", 1.5), ("9", 1.8), ("S", 0.8), ("s", 0.8)],
    "6": [("5", 1.5), ("0", 1.8), ("8", 1.8)],
    "9": [("3", 1.5), ("8", 1.5), ("7", 1.8), ("4", 2.0)],
    "2": [("3", 1.8), ("7", 2.0)],
    "4": [("1", 1.8), ("9", 2.0)],
}


class MathValidator:
    """2D Constraint-Based Mathematical Validation and Multi-Variable Self-Healing Engine."""

    @classmethod
    def detect_table_scale_multiplier(cls, table: TableData) -> Tuple[Decimal, Optional[str]]:
        header_text = " ".join(table.headers).lower()

        if table.rows and len(table.rows) > 0:
            first_row_text = " ".join(c.normalized_text for c in table.rows[0]).lower()
            header_text = f"{header_text} {first_row_text}"

        for pattern, mult, unit_name in HEADER_SCALE_PATTERNS:
            if pattern.search(header_text):
                logger.info("Detected table financial unit multiplier: %s (%s)", unit_name, mult)
                return mult, unit_name

        return Decimal("1"), None

    @classmethod
    def sanitize_financial_string(
        cls, text: str, default_multiplier: Decimal = Decimal("1")
    ) -> Optional[Decimal]:
        """Cleans Indian currency strings, parenthesized negatives (e.g. '(100.00)'), and Indian comma formatting."""
        if not text:
            return None

        # Convert Devanagari numerals to ASCII digits
        t = text.translate(DEV_TO_ASCII_DIGITS).strip().lower()

        # Check for parenthesized negative: (100.00) or ( 5,000.00 )
        is_negative = False
        if re.search(r"^\(.*\)$", t) or re.search(r"^-\s*", t):
            is_negative = True
            t = t.strip("()- ")

        # Check for explicit Indian scale words in cell text
        explicit_multiplier = None
        for word, mult in INDIAN_SCALE_MULTIPLIERS.items():
            if word in t:
                explicit_multiplier = mult
                t = t.replace(word, "").strip()
                break

        # Remove currency symbols, formatting commas, slashes, dashes, quotes
        cleaned = re.sub(r"[₹\$\£\€]|rs\.?|inr|[\,\/\-\'\"\s]", "", t)

        match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if not match:
            return None

        try:
            raw_num = Decimal(match.group(0))
            if is_negative:
                raw_num = -abs(raw_num)
            multiplier = explicit_multiplier if explicit_multiplier is not None else default_multiplier
            return raw_num * multiplier
        except (InvalidOperation, ValueError):
            return None

    @classmethod
    def find_total_in_table(
        cls, table: TableData
    ) -> Tuple[Optional[Decimal], Optional[int], Optional[int]]:
        total_keywords = ("योग", "कुल", "कुल योग", "महायोग", "total", "grand total", "sum")

        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row):
                norm = cell.normalized_text.strip().lower()
                if any(kw in norm for kw in total_keywords):
                    for num_c_idx in range(len(row) - 1, -1, -1):
                        num_cell = row[num_c_idx]
                        val = cls.sanitize_financial_string(
                            num_cell.normalized_text, default_multiplier=table.unit_multiplier
                        )
                        if val is not None:
                            return val, r_idx, num_c_idx

        return None, None, None

    @classmethod
    def validate_horizontal_constraints(cls, table: TableData, tolerance: Decimal = Decimal("0.01")) -> bool:
        """Validates 2D horizontal balance equations (e.g. Sanctioned - Expenditure = Balance)."""
        if not table.headers or not table.rows:
            return True

        headers_lower = [h.lower() for h in table.headers]
        sanctioned_col = -1
        expenditure_col = -1
        balance_col = -1

        for idx, h in enumerate(headers_lower):
            if any(k in h for k in ("स्वीकृत", "आवंटन", "sanctioned", "allotted", "provision")):
                sanctioned_col = idx
            elif any(k in h for k in ("व्यय", "expenditure", "spent")):
                expenditure_col = idx
            elif any(k in h for k in ("अवशेष", "balance", "remaining", "बचत")):
                balance_col = idx

        if sanctioned_col >= 0 and expenditure_col >= 0 and balance_col >= 0:
            for row in table.rows:
                if len(row) > max(sanctioned_col, expenditure_col, balance_col):
                    v_sanc = row[sanctioned_col].numeric_value
                    v_exp = row[expenditure_col].numeric_value
                    v_bal = row[balance_col].numeric_value

                    if v_sanc is not None and v_exp is not None and v_bal is not None:
                        expected_bal = v_sanc - v_exp
                        if abs(expected_bal - v_bal) > tolerance:
                            logger.info(
                                "Horizontal balance mismatch in row: %s - %s = %s != %s",
                                v_sanc, v_exp, expected_bal, v_bal
                            )
                            return False
        return True

    @classmethod
    def generate_digit_confusion_candidates(cls, raw_str: str, multiplier: Decimal) -> List[Tuple[Decimal, float]]:
        """Generates alternative numeric values based on digit confusion matrix with penalty scores."""
        clean = re.sub(r"[^\d\.]", "", raw_str)
        if not clean:
            return []

        candidates = []
        # Single digit mutations
        for idx, ch in enumerate(clean):
            if ch in DIGIT_CONFUSION_MAP:
                for alt_ch, penalty in DIGIT_CONFUSION_MAP[ch]:
                    mutated = clean[:idx] + alt_ch + clean[idx + 1:]
                    try:
                        val = Decimal(mutated) * multiplier
                        candidates.append((val, penalty))
                    except Exception:
                        pass
        return candidates

    @classmethod
    def validate_table_sums(
        cls,
        table: TableData,
        tolerance: Decimal = Decimal("0.01"),
        re_ocr_callback: Optional[Callable[[BoundingBox], Optional[str]]] = None,
    ) -> TableData:
        """Validates 2D vertical sums and horizontal equations with multi-variable confusion matrix solving."""
        multiplier, unit_name = cls.detect_table_scale_multiplier(table)
        table.unit_multiplier = multiplier
        table.unit_name = unit_name

        for row in table.rows:
            for cell in row:
                if cell.numeric_value is None:
                    cell.numeric_value = cls.sanitize_financial_string(
                        cell.normalized_text, default_multiplier=table.unit_multiplier
                    )

        declared_total, total_row_idx, amount_col_idx = cls.find_total_in_table(table)
        table.declared_total = declared_total

        if declared_total is None or total_row_idx is None or amount_col_idx is None:
            table.is_mathematically_valid = True
            return table

        row_values: List[Tuple[int, TableCell, Decimal]] = []
        for r_idx, row in enumerate(table.rows):
            if r_idx == total_row_idx:
                continue
            if amount_col_idx < len(row):
                cell = row[amount_col_idx]
                val = cell.numeric_value
                if val is not None:
                    row_values.append((r_idx, cell, val))

        computed_sum = sum((v for _, _, v in row_values), Decimal("0"))
        table.computed_total = computed_sum
        delta = abs(computed_sum - declared_total)
        table.validation_error_delta = delta

        # Check 1: Initial verification
        if delta <= tolerance:
            table.is_mathematically_valid = cls.validate_horizontal_constraints(table, tolerance)
            return table

        logger.warning(
            "Table verification mismatch: computed (%s) != declared (%s) [Delta: %s, Unit: %s]",
            computed_sum, declared_total, delta, table.unit_name or "units"
        )
        table.is_mathematically_valid = False

        # Check 2: Re-OCR Callback on Low-Confidence Cells
        if re_ocr_callback:
            ambiguous_candidates = sorted(
                [(r_idx, cell) for r_idx, cell, _ in row_values if cell.bbox is not None and cell.confidence < 0.95],
                key=lambda x: x[1].confidence,
            )

            for r_idx, cell in ambiguous_candidates:
                if cell.bbox is None:
                    continue
                re_scanned = re_ocr_callback(cell.bbox)
                if re_scanned:
                    new_val = cls.sanitize_financial_string(re_scanned, default_multiplier=table.unit_multiplier)
                    if new_val is not None:
                        cell.raw_text = re_scanned
                        cell.normalized_text = re_scanned
                        cell.numeric_value = new_val
                        cell.confidence = 1.0

            # Recompute after re-OCR
            new_computed = sum(
                (r[amount_col_idx].numeric_value or Decimal("0")
                 for i, r in enumerate(table.rows) if i != total_row_idx and amount_col_idx < len(r)),
                Decimal("0"),
            )
            new_delta = abs(new_computed - declared_total)
            if new_delta <= tolerance:
                logger.info("Self-healing successful via re-OCR! Mathematical validation restored.")
                table.computed_total = new_computed
                table.validation_error_delta = new_delta
                table.is_mathematically_valid = True
                return table

        # Check 3: Multi-Variable 2D Constraint Solving via Digit Confusion Matrix
        # Target: find cell values such that sum(rows) == declared_total with minimum penalty
        low_conf_cells = [
            (r_idx, cell) for r_idx, cell, _ in row_values
            if cell.confidence < 0.99 or (re_ocr_callback is None and len(row_values) <= 10)
        ][:4]  # Limit to top 4 ambiguous cells for combinatorial tractability

        if low_conf_cells:
            candidate_lists = []
            for r_idx, cell in low_conf_cells:
                alts = cls.generate_digit_confusion_candidates(cell.raw_text, table.unit_multiplier)
                # Include original as 0 penalty
                if cell.numeric_value is not None:
                    alts.append((cell.numeric_value, 0.0))
                candidate_lists.append(alts)

            best_solution = None
            min_penalty = float("inf")

            for combo in itertools.product(*candidate_lists):
                total_penalty = sum(p for _, p in combo)
                # Compute trial sum using (row_idx, col_idx) coordinates
                trial_vals = {(r_idx, cell.col_idx): val for (r_idx, cell), (val, _) in zip(low_conf_cells, combo)}
                trial_sum = sum(
                    trial_vals.get((i, amount_col_idx), r[amount_col_idx].numeric_value or Decimal("0"))
                    for i, r in enumerate(table.rows) if i != total_row_idx and amount_col_idx < len(r)
                )
                if abs(trial_sum - declared_total) <= tolerance:
                    if total_penalty < min_penalty:
                        min_penalty = total_penalty
                        best_solution = (trial_vals, trial_sum)

            if best_solution is not None and min_penalty <= 4.0:
                trial_vals, resolved_sum = best_solution
                for (r_idx, c_idx), new_v in trial_vals.items():
                    if r_idx < len(table.rows) and c_idx < len(table.rows[r_idx]):
                        table.rows[r_idx][c_idx].numeric_value = new_v
                table.computed_total = resolved_sum
                table.validation_error_delta = Decimal("0")
                table.is_mathematically_valid = True
                logger.info("Multi-variable 2D constraint solver resolved table reconciliation (Penalty: %.2f)", min_penalty)
                return table

        return table
