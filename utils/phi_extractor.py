"""
utils/phi_extractor.py

Regex-based PHI detection and BIO annotation for clinical text.
Handles: NAME, DOCTOR, HOSPITAL, LOCATION, DATE, PHONE, EMAIL, ID
"""

import re
from typing import List, Tuple, Dict

# ─── Compiled regex patterns ──────────────────────────────────────────────────

# Dates: 12 March 2024, 12/03/2024, March 12, 2024, 2024-03-12
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[\s/-](?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s/-]\d{2,4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)

# Phone numbers: +91-9876543210, (080) 1234-5678, 9876543210
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,6}\b"
)

# Email addresses
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Patient IDs: PT123456, MRN-00123, IP/2024/001
_ID_PATTERN = re.compile(
    r"\b(?:PT|MRN|IP|OP|REG|UHID|PID|CRN)[-/]?\d{4,10}\b",
    re.IGNORECASE,
)

# Doctor name following "Dr" prefix
_DOCTOR_PATTERN = re.compile(
    r"\bDr\.?\s+[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b"
)

# Patient name following "Patient Name:" or "Patient:" labels
_PATIENT_LABEL_PATTERN = re.compile(
    r"(?:Patient(?:\s+Name)?|Mr\.|Mrs\.|Ms\.|Pt\.)\s*:?\s*([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})"
)

# Hospital name following "Hospital:" or matching known patterns
_HOSPITAL_PATTERN = re.compile(
    r"\b[A-Z][a-zA-Z\s]+"
    r"(?:Hospital|Clinic|Medical Centre?|Health\s*Care|Institute|AIIMS|CMC|PGIMER)\b"
)

# Location: city after "in", "at", "from"
_LOCATION_CONTEXT = re.compile(
    r"(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


def extract_phi_from_text(text: str) -> List[Tuple[int, int, str]]:
    """
    Run all PHI regex patterns against text.
    Returns sorted list of (start, end, entity_type) non-overlapping spans.
    """
    spans = []

    for m in _DATE_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "DATE"))

    for m in _EMAIL_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "EMAIL"))

    for m in _PHONE_PATTERN.finditer(text):
        # Sanity check: must have at least 7 digits
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 7:
            spans.append((m.start(), m.end(), "PHONE"))

    for m in _ID_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "ID"))

    for m in _DOCTOR_PATTERN.finditer(text):
        # Store the full match but tag as DOCTOR
        spans.append((m.start(), m.end(), "DOCTOR"))

    for m in _PATIENT_LABEL_PATTERN.finditer(text):
        # Group 1 is the name value
        g = m.group(1)
        s = m.start(1)
        spans.append((s, s + len(g), "NAME"))

    for m in _HOSPITAL_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "HOSPITAL"))

    for m in _LOCATION_CONTEXT.finditer(text):
        g = m.group(1)
        s = m.start(1)
        spans.append((s, s + len(g), "LOCATION"))

    # Remove overlapping spans (keep longest)
    spans = _resolve_overlaps(spans)
    return sorted(spans, key=lambda x: x[0])


def _resolve_overlaps(spans):
    """Keep the longest span when multiple spans overlap."""
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    result = []
    last_end = -1
    for start, end, etype in spans:
        if start >= last_end:
            result.append((start, end, etype))
            last_end = end
    return result


def _tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text and return (token, start, end) tuples.
    """
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


def annotate_tokens_bio(text: str) -> Tuple[List[str], List[str]]:
    """
    Tokenize text and assign BIO tags based on regex PHI detection.

    Returns:
        tokens: list of token strings
        tags: list of BIO tag strings
    """
    raw_tokens = _tokenize(text)
    if not raw_tokens:
        return [], []

    phi_spans = extract_phi_from_text(text)
    tags = []

    for tok, tok_start, tok_end in raw_tokens:
        label = "O"
        for span_start, span_end, etype in phi_spans:
            if tok_start >= span_start and tok_end <= span_end:
                label = f"B-{etype}" if tok_start == span_start else f"I-{etype}"
                break
        tags.append(label)

    tokens = [t[0] for t in raw_tokens]
    return tokens, tags


def redact_text(text: str, replacement_template: str = "[{entity_type}]") -> str:
    """
    Replace all PHI spans in text with redaction placeholders.

    Args:
        text: Input clinical text.
        replacement_template: Template with {entity_type} placeholder.

    Returns:
        Redacted text string.
    """
    spans = extract_phi_from_text(text)
    # Replace from right to left to preserve character positions
    for start, end, etype in reversed(spans):
        replacement = replacement_template.format(entity_type=etype)
        text = text[:start] + replacement + text[end:]
    return text
