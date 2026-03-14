"""
data_generation/generate_dataset_from_reports.py

Processes PDF medical reports from reports/ directory:
  1. Extract text with pdfminer.six
  2. Detect PHI fields with regex rules
  3. Convert to BIO-tagged CoNLL sentences
  4. Append to existing train/dev/test splits
"""

import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    REPORTS_DIR,
    TRAIN_FILE,
    DEV_FILE,
    TEST_FILE,
    TRAIN_RATIO,
    DEV_RATIO,
)
from utils.pdf_parser import extract_text_from_pdf
from utils.phi_extractor import extract_phi_from_text, annotate_tokens_bio


def read_reports(reports_dir: Path):
    """Yield (filename, raw_text) for each PDF in reports/."""
    pdfs = list(reports_dir.glob("**/*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {reports_dir}. Skipping real-report generation.")
        return
    for pdf_path in pdfs:
        print(f"  Processing: {pdf_path.name}")
        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                yield pdf_path.name, text
        except Exception as e:
            print(f"    ⚠  Failed to parse {pdf_path.name}: {e}")


def split_into_sentences(text: str):
    """Split clinical text into sentence-sized chunks."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def process_report(text: str):
    """
    Extract PHI entities and clinical sentences from a report.
    Returns list of (tokens, tags) tuples.
    """
    sentences = split_into_sentences(text)
    samples = []
    for sentence in sentences:
        tokens, tags = annotate_tokens_bio(sentence)
        if tokens:
            samples.append((tokens, tags))
    return samples


def append_conll(samples, filepath: Path):
    """Append samples to an existing CoNLL file."""
    with open(filepath, "a", encoding="utf-8") as f:
        for tokens, tags in samples:
            for tok, tag in zip(tokens, tags):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def main():
    print(f"Scanning reports directory: {REPORTS_DIR}")
    all_samples = []

    for fname, text in read_reports(REPORTS_DIR):
        samples = process_report(text)
        print(f"    → Extracted {len(samples)} annotated sentences from {fname}")
        all_samples.extend(samples)

    if not all_samples:
        print("No real-report samples generated.")
        return

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)

    train = all_samples[:n_train]
    dev   = all_samples[n_train:n_train + n_dev]
    test  = all_samples[n_train + n_dev:]

    append_conll(train, TRAIN_FILE)
    append_conll(dev,   DEV_FILE)
    append_conll(test,  TEST_FILE)

    print(f"\nAppended from real reports:")
    print(f"  Train : +{len(train):,}")
    print(f"  Dev   : +{len(dev):,}")
    print(f"  Test  : +{len(test):,}")


if __name__ == "__main__":
    main()
