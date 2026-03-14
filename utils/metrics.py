"""
utils/metrics.py

Entity-level NER evaluation metrics (precision, recall, F1).
Implements strict span-based evaluation per CoNLL convention.
"""

from typing import List, Dict, Tuple
from collections import defaultdict


def _extract_entities(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extract entity spans from a BIO tag sequence.
    Returns list of (start_idx, end_idx_exclusive, entity_type).
    """
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            entities.append((i, j, etype))
            i = j
        else:
            i += 1
    return entities


def compute_entity_metrics(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute entity-level precision, recall, and F1 per entity type and overall.

    Args:
        true_tags_list: List of gold tag sequences.
        pred_tags_list: List of predicted tag sequences.

    Returns:
        Dict mapping entity_type (and "overall") → {"precision", "recall", "f1", "support"}.
    """
    # Per-type counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
        true_entities = set(_extract_entities(true_tags))
        pred_entities = set(_extract_entities(pred_tags))

        for span in pred_entities:
            etype = span[2]
            if span in true_entities:
                tp[etype] += 1
            else:
                fp[etype] += 1

        for span in true_entities:
            etype = span[2]
            support[etype] += 1
            if span not in pred_entities:
                fn[etype] += 1

    results = {}
    all_types = set(list(tp.keys()) + list(fn.keys()))

    for etype in sorted(all_types):
        p = tp[etype] / (tp[etype] + fp[etype] + 1e-9)
        r = tp[etype] / (tp[etype] + fn[etype] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        results[etype] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "support":   support[etype],
        }

    # Overall
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    p = total_tp / (total_tp + total_fp + 1e-9)
    r = total_tp / (total_tp + total_fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    results["overall"] = {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f, 4),
        "support":   sum(support.values()),
    }
    return results


def format_metrics_report(metrics: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print metrics table."""
    lines = [
        f"{'Entity':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 60,
    ]
    for etype, vals in metrics.items():
        if etype == "overall":
            continue
        lines.append(
            f"{etype:<15} {vals['precision']:>10.4f} {vals['recall']:>10.4f}"
            f" {vals['f1']:>10.4f} {vals['support']:>10}"
        )
    lines.append("-" * 60)
    ov = metrics.get("overall", {})
    lines.append(
        f"{'OVERALL':<15} {ov.get('precision',0):>10.4f} {ov.get('recall',0):>10.4f}"
        f" {ov.get('f1',0):>10.4f} {ov.get('support',0):>10}"
    )
    return "\n".join(lines)
