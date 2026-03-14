"""
dataset/vocab_builder.py

Builds word vocabulary from CoNLL training data.
Saves vocab.txt for reproducibility.
"""

import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    TRAIN_FILE,
    VOCAB_FILE,
    TAG_FILE,
    PAD_TOKEN,
    UNK_TOKEN,
    PAD_IDX,
    UNK_IDX,
    TAGS,
    TAG2IDX,
)


def load_conll(filepath: Path):
    """Load CoNLL file, return list of (tokens, tags) sentence pairs."""
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tok, tag = parts
                    tokens.append(tok)
                    tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def build_vocab(train_file: Path, min_freq: int = 1) -> Dict[str, int]:
    """
    Build word→index vocabulary from training data.

    Args:
        train_file: Path to train.txt in CoNLL format.
        min_freq: Minimum token frequency to include.

    Returns:
        word2idx dict.
    """
    counter = Counter()
    sentences = load_conll(train_file)
    for tokens, _ in sentences:
        counter.update(tokens)

    # Special tokens first
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    idx = 2
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in word2idx:
            word2idx[word] = idx
            idx += 1

    return word2idx


def save_vocab(word2idx: Dict[str, int], vocab_file: Path):
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w", encoding="utf-8") as f:
        for word, idx in sorted(word2idx.items(), key=lambda x: x[1]):
            f.write(f"{word}\t{idx}\n")
    print(f"Saved vocabulary ({len(word2idx):,} tokens) → {vocab_file}")


def load_vocab(vocab_file: Path) -> Dict[str, int]:
    word2idx = {}
    with open(vocab_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                word, idx = line.rsplit("\t", 1)
                word2idx[word] = int(idx)
    return word2idx


def save_tags(tag2idx: Dict[str, int], tag_file: Path):
    tag_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tag_file, "w", encoding="utf-8") as f:
        for tag, idx in sorted(tag2idx.items(), key=lambda x: x[1]):
            f.write(f"{tag}\t{idx}\n")


def main():
    print(f"Building vocabulary from {TRAIN_FILE}...")
    word2idx = build_vocab(TRAIN_FILE)
    save_vocab(word2idx, VOCAB_FILE)
    save_tags(TAG2IDX, TAG_FILE)
    print(f"Tag set ({len(TAG2IDX)} tags): {list(TAG2IDX.keys())}")


if __name__ == "__main__":
    main()
