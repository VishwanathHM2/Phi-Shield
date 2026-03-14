"""
utils/tokenizer.py

Clinical text tokenizer with simple whitespace + punctuation splitting.
"""

import re
from typing import List


def tokenize(text: str) -> List[str]:
    """
    Tokenize clinical text.
    Splits on whitespace while preserving hyphenated/compound tokens
    (e.g., "48-year-old", "eGFR", "HbA1c").
    Punctuation adjacent to words is separated.
    """
    # Separate trailing/leading punctuation except hyphens and slashes inside tokens
    tokens = re.findall(r"\w+(?:[-/]\w+)*|[^\w\s]", text)
    return [t for t in tokens if t.strip()]


def detokenize(tokens: List[str]) -> str:
    """
    Naive detokenization: join tokens with spaces,
    attach punctuation to previous token.
    """
    if not tokens:
        return ""
    result = tokens[0]
    for tok in tokens[1:]:
        if tok in {",", ".", "!", "?", ";", ":", ")", "]", "}"}:
            result += tok
        else:
            result += " " + tok
    return result
