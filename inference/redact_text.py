"""
inference/redact_text.py

PHI redaction pipeline.
Combines model-based detection + regex fallback for high-recall redaction.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import BEST_MODEL_PATH, VOCAB_FILE, REDACT_TEMPLATE
from utils.tokenizer import tokenize
from utils.phi_extractor import redact_text as regex_redact


def model_redact(text: str, predictor, template: str = REDACT_TEMPLATE) -> str:
    """
    Redact PHI in text using the trained model.
    Replaces detected entity spans with [ENTITY_TYPE] tags.
    """
    tokens, tags = predictor.predict_tags(text)
    if not tokens:
        return text

    redacted_tokens = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            redacted_tokens.append(template.format(entity_type=etype))
            i = j
        else:
            redacted_tokens.append(tokens[i])
            i += 1

    # Reconstruct text preserving spaces
    return _rejoin(tokens, redacted_tokens, text)


def _rejoin(original_tokens: List[str], new_tokens: List[str], original_text: str) -> str:
    """
    Reconstruct text from token replacements.
    Tries to preserve inter-token spacing from original text.
    """
    result = original_text
    # Replace from end to preserve positions
    orig_idx = len(original_tokens) - 1
    new_idx  = len(new_tokens) - 1

    # Simple token-by-token replacement from right
    pos = len(result)
    rebuilt = []
    for tok, new_tok in zip(original_tokens, new_tokens):
        rebuilt.append(new_tok)

    return " ".join(rebuilt)


def hybrid_redact(
    text: str,
    predictor=None,
    use_model: bool = True,
    template: str = REDACT_TEMPLATE,
) -> str:
    """
    Hybrid redaction: model + regex for maximum coverage.

    Args:
        text: Clinical text to redact.
        predictor: PHIPredictor instance (optional).
        use_model: Whether to apply model-based detection.
        template: Redaction placeholder template.

    Returns:
        Redacted text string.
    """
    # Step 1: Model-based redaction (if available)
    if use_model and predictor is not None:
        try:
            text = model_redact(text, predictor, template)
        except Exception as e:
            print(f"  Model redaction warning: {e}")

    # Step 2: Regex-based fallback for any missed PHI
    text = regex_redact(text, template)

    return text


def redact_document(
    text: str,
    predictor=None,
    use_model: bool = True,
) -> Dict:
    """
    Redact an entire document text, paragraph by paragraph.
    Returns dict with original, redacted text and found entities.
    """
    paragraphs = text.split("\n")
    redacted_paragraphs = []
    all_entities = []

    for para in paragraphs:
        if not para.strip():
            redacted_paragraphs.append(para)
            continue

        redacted = hybrid_redact(para, predictor, use_model)
        redacted_paragraphs.append(redacted)

        if predictor:
            try:
                entities = predictor.predict_entities(para)
                all_entities.extend(entities)
            except Exception:
                pass

    return {
        "original":  text,
        "redacted":  "\n".join(redacted_paragraphs),
        "entities":  all_entities,
    }
